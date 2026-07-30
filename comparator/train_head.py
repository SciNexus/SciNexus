from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SciNexus import paths as P
from SciNexus.comparator.comparator import SYSTEM_PROMPT, USER_TEMPLATE
from SciNexus.comparator.paths import (
    base_model_path,
    default_head_output,
    default_test_data,
    init_lora_path,
)

@dataclass(frozen=True)
class LabelExample:
    text: str
    label: int

class TextDataset(Dataset):
    def __init__(self, examples: list[LabelExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LabelExample:
        return self.examples[index]

def load_label_examples(paths: list[Path], tokenizer) -> list[LabelExample]:
    examples: list[LabelExample] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                winner = rec.get("true_winner")
                if winner not in {"Hypothesis A", "Hypothesis B"}:
                    continue
                hyp_a = rec.get("idea_A_json", "")
                hyp_b = rec.get("idea_B_json", "")
                if not hyp_a or not hyp_b:
                    continue
                user_prompt = USER_TEMPLATE.format(
                    current_method="",
                    limitation="",
                    hyp_a=hyp_a[:1500],
                    hyp_b=hyp_b[:1500],
                )
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                examples.append(LabelExample(text=text, label=0 if winner == "Hypothesis A" else 1))
    return examples

def split_label_examples(
    examples: list[LabelExample],
    val_frac: float,
    seed: int,
) -> tuple[list[LabelExample], list[LabelExample]]:
    rng = random.Random(seed)
    by_label: dict[int, list[LabelExample]] = {0: [], 1: []}
    for ex in examples:
        by_label[ex.label].append(ex)
    train: list[LabelExample] = []
    val: list[LabelExample] = []
    for label_examples in by_label.values():
        rng.shuffle(label_examples)
        n_val = max(1, int(round(len(label_examples) * val_frac)))
        val.extend(label_examples[:n_val])
        train.extend(label_examples[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

def collate_text(batch: list[LabelExample]) -> tuple[list[str], torch.Tensor]:
    return [ex.text for ex in batch], torch.tensor([ex.label for ex in batch], dtype=torch.long)

def load_frozen_model(base_model: str | None, lora_path: str | None):
    base_path = base_model or base_model_path()
    lora = lora_path if lora_path is not None else init_lora_path()
    if not lora:
        raise ValueError("LoRA checkpoint required. Set COMPARATOR_INIT_LORA or pass --init-lora.")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=torch.cuda.is_available(),
    ).to(device)
    model = PeftModel.from_pretrained(base, lora, is_trainable=False).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer

def extract_features(model, tokenizer, examples, batch_size, max_length):
    dataset = TextDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_text)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    device = model.device
    with torch.no_grad():
        for step, (texts, y) in enumerate(loader, start=1):
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[-1][:, -1, :].float().cpu()
            features.append(hidden)
            labels.append(y)
            if step % 20 == 0:
                print(f"  extracted {min(step * batch_size, len(examples))}/{len(examples)}")
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

def evaluate_head(head: nn.Linear, features: torch.Tensor, labels: torch.Tensor):
    head.eval()
    with torch.no_grad():
        logits = head(features)
        loss = F.cross_entropy(logits, labels).item()
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc

def train_head(train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    head = nn.Linear(train_x.shape[1], 2)
    counts = torch.bincount(train_y, minlength=2).float()
    weights = counts.sum() / (2.0 * counts.clamp_min(1.0))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    dataset = TensorDataset(train_x, train_y)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    best_state = None
    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_seen = 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            loss = F.cross_entropy(logits, y, weight=weights)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
            total_seen += x.shape[0]
        train_loss, train_acc = evaluate_head(head, train_x, train_y)
        val_loss, val_acc = evaluate_head(head, val_x, val_y)
        print(
            f"epoch={epoch:02d} batch_loss={total_loss / max(1, total_seen):.4f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return head

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=default_test_data())
    parser.add_argument("--output", default=default_head_output())
    parser.add_argument("--init-lora", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(P.resolve_pkg_path_str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    init_lora = args.init_lora or init_lora_path()
    base_model = args.base_model or base_model_path()

    print("loading frozen comparator backbone...")
    model, tokenizer = load_frozen_model(base_model, init_lora)
    examples = load_label_examples([P.resolve_pkg_path(p) for p in args.data], tokenizer)
    train_examples, val_examples = split_label_examples(examples, args.val_frac, args.seed)
    print(
        f"examples={len(examples)} train={len(train_examples)} val={len(val_examples)} "
        f"train_A={sum(e.label == 0 for e in train_examples)} train_B={sum(e.label == 1 for e in train_examples)}"
    )
    print("extracting train features...")
    train_x, train_y = extract_features(
        model, tokenizer, train_examples, args.batch_size, args.max_length
    )
    print("extracting val features...")
    val_x, val_y = extract_features(
        model, tokenizer, val_examples, args.batch_size, args.max_length
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("training classification head...")
    head = train_head(
        train_x,
        train_y,
        val_x,
        val_y,
        args.epochs,
        args.head_batch_size,
        args.lr,
        args.seed,
    )
    val_loss, val_acc = evaluate_head(head, val_x, val_y)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": train_x.shape[1],
            "labels": {"Hypothesis A": 0, "Hypothesis B": 1},
            "val_loss": val_loss,
            "val_acc": val_acc,
            "data": [str(p) for p in args.data],
        },
        output,
    )
    print(f"saved {output}")
    print(f"final val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

if __name__ == "__main__":
    main()
