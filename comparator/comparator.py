from __future__ import annotations
import re
from typing import Optional
import torch
from torch import nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from SciNexus.config import ModelConfig
SYSTEM_PROMPT = (
    "You are an expert scientific evaluator. You always respond in a structured format "
    "using <think> and <decision> tags. You should give the winner in the <decision> tag. "
    "Such as <decision>\nWinner: Hypothesis A</decision>"
)
USER_TEMPLATE = """\
Evaluate these two scientific hypotheses based on their theoretical merit.
**CRITICAL EVALUATION CONSTRAINTS:**
1. **Focus Exclusively on Theoretical Merit:** Your task is to evaluate the \
theoretical soundness, logical consistency, and explanatory power of the core \
ideas and mechanisms proposed.
2. **Equal Maturity Assumption:** Treat both Hypothesis A and Hypothesis B as \
purely conceptual proposals of equal testing maturity. Compare the brilliance \
and plausibility of the *ideas themselves*, not their execution.
Scientific Problem Context:
Scientific Context: {scientific_context}
Hypothesis A:
{hyp_a}
Hypothesis B:
{hyp_b}
Which hypothesis better addresses the limitation above? Respond with <think> \
reasoning then <decision> Winner: Hypothesis A or B </decision>."""
class HypothesisComparator:
    def __init__(self, config: ModelConfig):
        self._max_new_tokens = config.comparator_max_new_tokens
        self._max_input_length = config.comparator_max_input_length
        self._head_temperature = float(
            getattr(config, "comparator_head_temperature", 2.0)
        )
        self._head_position_swap = bool(
            getattr(config, "comparator_head_position_swap", True)
        )
        self._cache: dict[str, int] = {}
        self._score_cache: dict[str, float] = {}
        self._scores_cache: dict[str, tuple[float, float, float]] = {}
        self._model, self._tokenizer = self._load_model(
            config.comparator_base_model_path,
            config.comparator_lora_path,
        )
        self._reward_head = self._load_reward_head(
            getattr(config, "comparator_head_path", "")
        )
    @staticmethod
    def _load_model(
        base_model_path: str,
        lora_adapter_path: str,
    ) -> tuple:
        print(f"[Comparator] Loading base model: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float16 if load_device.type == "cuda" else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=torch.cuda.is_available(),
        ).to(load_device)
        print(f"[Comparator] Mounting LoRA adapter: {lora_adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            is_trainable=False,
        )
        model.eval()
        print("[Comparator] Model ready.")
        return model, tokenizer
    def _load_reward_head(self, head_path: str) -> nn.Linear:
        model_config = getattr(self._model, "config", None)
        hidden_size = getattr(model_config, "hidden_size", None)
        if hidden_size is None and hasattr(self._model, "get_base_model"):
            base_config = getattr(self._model.get_base_model(), "config", None)
            hidden_size = getattr(base_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer comparator hidden size for reward head")
        hidden_size = int(hidden_size)
        dtype = next(self._model.parameters()).dtype
        head = nn.Linear(hidden_size, 2)
        if head_path:
            try:
                state = torch.load(head_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                head.load_state_dict(state)
                print(f"[Comparator] Loaded reward head: {head_path}")
            except FileNotFoundError:
                print(f"[Comparator] Reward head not found, initialized new: {head_path}")
        else:
            print("[Comparator] No reward head path set; initialized new head.")
        head.to(device=self._model.device, dtype=dtype)
        head.eval()
        for param in head.parameters():
            param.requires_grad_(False)
        return head
    def compare(
        self,
        task: dict,
        hyp_a: str,
        hyp_b: str,
        use_cache: bool = True,
    ) -> int:
        if not hyp_a or not hyp_b:
            return 0
        cache_key = self._make_key(task, hyp_a, hyp_b)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        user_prompt = USER_TEMPLATE.format(
            scientific_context=task.get("scientific_context", ""),
            hyp_a=hyp_a,
            hyp_b=hyp_b,
        )
        response = self._run_inference(user_prompt)
        winner = self._extract_prediction(response)
        score = {"Hypothesis A": 1, "Hypothesis B": -1}.get(winner, 0)
        if use_cache:
            self._cache[cache_key] = score
        return score
    def __call__(self, task: dict, hyp_a: str, hyp_b: str) -> int:
        return self.compare(task, hyp_a, hyp_b)
    def compare_logits(
        self,
        task: dict,
        hyp_a: str,
        hyp_b: str,
    ) -> tuple[float, float]:
        if not hyp_a or not hyp_b:
            return 0.0, 0.0
        user_prompt = self._format_user_prompt(task, hyp_a, hyp_b)
        logits = self._run_reward_head(user_prompt)
        return float(logits[0]), float(logits[1])
    def compare_scores(
        self,
        task: dict,
        hyp_a: str,
        hyp_b: str,
        use_cache: bool = True,
        position_swap: Optional[bool] = None,
    ) -> tuple[float, float, float]:
        if not hyp_a or not hyp_b:
            return 0.0, 0.0, 0.0
        do_swap = self._head_position_swap if position_swap is None else position_swap
        cache_key = f"scores||swap={do_swap}||{self._make_key(task, hyp_a, hyp_b)}"
        if use_cache and cache_key in self._scores_cache:
            cached = self._scores_cache[cache_key]
            return float(cached[0]), float(cached[1]), float(cached[2])
        za, zb = self.compare_logits(task, hyp_a, hyp_b)
        score_a, score_b = za, zb
        margin = za - zb
        if do_swap:
            za_s, zb_s = self.compare_logits(task, hyp_b, hyp_a)
            score_a = 0.5 * (za + zb_s)
            score_b = 0.5 * (zb + za_s)
            margin = 0.5 * (margin - (za_s - zb_s))
        out = (float(score_a), float(score_b), float(margin))
        if use_cache:
            self._scores_cache[cache_key] = out
        return out
    def compare_margin(
        self,
        task: dict,
        hyp_a: str,
        hyp_b: str,
        use_cache: bool = True,
        position_swap: Optional[bool] = None,
    ) -> float:
        return self.compare_scores(
            task, hyp_a, hyp_b, use_cache=use_cache, position_swap=position_swap
        )[2]
    def compare_score(
        self,
        task: dict,
        hyp_a: str,
        hyp_b: str,
        use_cache: bool = True,
        position_swap: Optional[bool] = None,
    ) -> float:
        margin = self.compare_margin(
            task, hyp_a, hyp_b, use_cache=use_cache, position_swap=position_swap
        )
        return float(torch.tanh(torch.tensor(margin / self._head_temperature)).item())
    def _build_chat_text(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    @staticmethod
    def _format_user_prompt(task: dict, hyp_a: str, hyp_b: str) -> str:
        return USER_TEMPLATE.format(
            scientific_context=task.get("scientific_context", ""),
            hyp_a=hyp_a,
            hyp_b=hyp_b,
        )
    def _run_reward_head(self, user_prompt: str) -> torch.Tensor:
        text = self._build_chat_text(user_prompt)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_length,
        ).to(self._model.device)
        with torch.no_grad():
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1][:, -1, :]
            logits = self._reward_head(hidden)[0].float().cpu()
        return logits
    def _run_inference(self, user_prompt: str) -> str:
        text = self._build_chat_text(user_prompt)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_length,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated_ids = output[0][input_len:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)
    def run_inference_batch(self, user_prompts: list[str]) -> list[str]:
        texts = [self._build_chat_text(p) for p in user_prompts]
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_input_length,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        responses = []
        for seq in outputs:
            generated_ids = seq[input_len:]
            responses.append(
                self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
        return responses
    def compare_batch(
        self,
        tasks: list[dict],
        hyps_a: list[str],
        hyps_b: list[str],
        use_cache: bool = True,
    ) -> list[int]:
        results: list[Optional[int]] = [None] * len(tasks)
        pending_idx: list[int] = []
        pending_prompts: list[str] = []
        for k, (task, a, b) in enumerate(zip(tasks, hyps_a, hyps_b)):
            if not a or not b:
                results[k] = 0
                continue
            key = self._make_key(task, a, b)
            if use_cache and key in self._cache:
                results[k] = self._cache[key]
            else:
                pending_idx.append(k)
                pending_prompts.append(
                    USER_TEMPLATE.format(
                        scientific_context=task.get("scientific_context", ""),
                        hyp_a=a,
                        hyp_b=b,
                    )
                )
        if pending_prompts:
            responses = self.run_inference_batch(pending_prompts)
            for k, response in zip(pending_idx, responses):
                winner = self._extract_prediction(response)
                score = {"Hypothesis A": 1, "Hypothesis B": -1}.get(winner, 0)
                if use_cache:
                    key = self._make_key(tasks[k], hyps_a[k], hyps_b[k])
                    self._cache[key] = score
                results[k] = score
        return results                              
    @staticmethod
    def _extract_prediction(response_text: str) -> str:
        decision_match = re.search(
            r'<decision>(.*?)</decision>',
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if decision_match:
            block = decision_match.group(1)
            winner_match = re.search(
                r'Winner\s*:\s*(Hypothesis\s*[AB])', block, re.IGNORECASE
            )
            if winner_match:
                raw = winner_match.group(1).strip()
                if re.search(r'[Aa]$', raw):
                    return "Hypothesis A"
                if re.search(r'[Bb]$', raw):
                    return "Hypothesis B"
        return "Unknown"
    @staticmethod
    def _make_key(task: dict, a: str, b: str) -> str:
        return f"str({task})||str({a})||str({b})"
    def rank_hypotheses(self, task: dict, hypotheses: list[str]) -> list[str]:
        from itertools import combinations
        scores = {i: 0 for i in range(len(hypotheses))}
        for i, j in combinations(range(len(hypotheses)), 2):
            r = self.compare(task, hypotheses[i], hypotheses[j])
            scores[i] += r
            scores[j] -= r
        ranked_idx = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [hypotheses[i] for i in ranked_idx]
