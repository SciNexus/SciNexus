from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from SciNexus import paths as P
from SciNexus.comparator.comparator import SYSTEM_PROMPT, USER_TEMPLATE
from SciNexus.comparator.paths import base_model_path, init_lora_path

@dataclass(frozen=True)
class FwciExample:
    text: str
    target: tuple[float, float]

class TextDataset(Dataset):
    def __init__(self, examples: list[FwciExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> FwciExample:
        return self.examples[index]

class ComparatorRegressionModel(nn.Module):
    def __init__(self, backbone: PeftModel, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1][:, -1, :].float()
        return self.head(hidden)

def build_chat_text(tokenizer, hyp_a: str, hyp_b: str) -> str:
    user_prompt = USER_TEMPLATE.format(
        current_method="",
        limitation="",
        hyp_a=hyp_a[:1500],
        hyp_b=hyp_b[:1500],
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0:
        return None
    return out

def load_fwci_examples(
    paths: list[Path],
    tokenizer,
    *,
    add_swaps: bool,
) -> list[FwciExample]:
    examples: list[FwciExample] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                hyp_a = rec.get("idea_A_json", "")
                hyp_b = rec.get("idea_B_json", "")
                fwci_a = _safe_float(rec.get("fwci_A"))
                fwci_b = _safe_float(rec.get("fwci_B"))
                if not hyp_a or not hyp_b or fwci_a is None or fwci_b is None:
                    continue
                target_a = math.log1p(fwci_a)
                target_b = math.log1p(fwci_b)
                examples.append(
                    FwciExample(
                        text=build_chat_text(tokenizer, hyp_a, hyp_b),
                        target=(target_a, target_b),
                    )
                )
                if add_swaps:
                    examples.append(
                        FwciExample(
                            text=build_chat_text(tokenizer, hyp_b, hyp_a),
                            target=(target_b, target_a),
                        )
                    )
    return examples

def split_examples(
    examples: list[FwciExample],
    val_frac: float,
    seed: int,
) -> tuple[list[FwciExample], list[FwciExample]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    return shuffled[n_val:], shuffled[:n_val]

def collate_fwci(batch: list[FwciExample]) -> tuple[list[str], torch.Tensor]:
    texts = [ex.text for ex in batch]
    targets = torch.tensor([ex.target for ex in batch], dtype=torch.float32)
    return texts, targets

def load_trainable_model(
    device: torch.device,
    *,
    base_model: str | None = None,
    init_lora: str | None = None,
) -> tuple[ComparatorRegressionModel, AutoTokenizer]:
    base_path = base_model or base_model_path()
    lora_path = init_lora if init_lora is not None else init_lora_path()
    if not lora_path:
        raise ValueError(
            "Initial LoRA checkpoint required. Set COMPARATOR_INIT_LORA or pass --init-lora."
        )
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=torch.cuda.is_available(),
    ).to(device)
    for param in base.parameters():
        param.requires_grad_(False)
    backbone = PeftModel.from_pretrained(
        base,
        lora_path,
        is_trainable=True,
    )
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    hidden_size = int(getattr(backbone.config, "hidden_size", 0))
    if hidden_size <= 0:
        hidden_size = int(getattr(backbone.get_base_model().config, "hidden_size"))
    model = ComparatorRegressionModel(backbone, hidden_size).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    return model, tokenizer

def normalize_targets(
    train_y_raw: torch.Tensor,
    *others: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], float, float]:
    flat = train_y_raw.reshape(-1)
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).clamp_min(1e-6).item())
    train_y = (train_y_raw - mean) / std
    normalized = [(y - mean) / std for y in others]
    return train_y, normalized, mean, std

def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((x @ y / denom).item())

def head_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_margin: float,
    lambda_rank: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    value_loss = F.smooth_l1_loss(pred, target)
    pred_margin = pred[:, 0] - pred[:, 1]
    target_margin = target[:, 0] - target[:, 1]
    margin_loss = F.smooth_l1_loss(pred_margin, target_margin)
    rank_target = (target_margin > 0).float()
    rank_weight = target_margin.abs().clamp_min(0.1)
    rank_loss = F.binary_cross_entropy_with_logits(
        pred_margin,
        rank_target,
        weight=rank_weight,
    )
    loss = value_loss + lambda_margin * margin_loss + lambda_rank * rank_loss
    return loss, {
        "value": float(value_loss.item()),
        "margin": float(margin_loss.item()),
        "rank": float(rank_loss.item()),
    }

@torch.no_grad()
def evaluate_fwci(
    model: ComparatorRegressionModel,
    tokenizer,
    examples: list[FwciExample],
    targets: torch.Tensor,
    batch_size: int,
    max_length: int,
    lambda_margin: float,
    lambda_rank: float,
    label: str,
) -> dict[str, float]:
    if not examples:
        return {
            "loss": 0.0,
            "value_loss": 0.0,
            "margin_loss": 0.0,
            "rank_loss": 0.0,
            "rank_acc": 0.0,
            "margin_mae": 0.0,
            "value_mae": 0.0,
            "corr_value": 0.0,
            "corr_margin": 0.0,
        }
    model.eval()
    dataset = TextDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fwci)
    device = next(model.parameters()).device
    preds: list[torch.Tensor] = []
    progress = tqdm(loader, desc=f"eval {label}", unit="batch", leave=False)
    for texts, _ in progress:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        pred = model(inputs["input_ids"], inputs["attention_mask"]).float().cpu()
        preds.append(pred)
    pred = torch.cat(preds, dim=0)
    targets = targets.cpu()
    loss, parts = head_loss(pred, targets, lambda_margin, lambda_rank)
    pred_margin = pred[:, 0] - pred[:, 1]
    target_margin = targets[:, 0] - targets[:, 1]
    rank_acc = ((pred_margin > 0) == (target_margin > 0)).float().mean().item()
    margin_mae = (pred_margin - target_margin).abs().mean().item()
    value_mae = (pred - targets).abs().mean().item()
    return {
        "loss": float(loss.item()),
        "value_loss": parts["value"],
        "margin_loss": parts["margin"],
        "rank_loss": parts["rank"],
        "rank_acc": float(rank_acc),
        "margin_mae": float(margin_mae),
        "value_mae": float(value_mae),
        "corr_value": pearson(pred, targets),
        "corr_margin": pearson(pred_margin, target_margin),
    }

def train_fwci_model(
    model: ComparatorRegressionModel,
    tokenizer,
    train_examples: list[FwciExample],
    train_y: torch.Tensor,
    val_examples: list[FwciExample],
    val_y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    max_length: int,
    lora_lr: float,
    head_lr: float,
    lambda_margin: float,
    lambda_rank: float,
    seed: int,
    skip_train_eval: bool = False,
    checkpoint_dir: Path | None = None,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> None:
    device = next(model.parameters()).device
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=0.01,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -1e9
    train_dataset = TextDataset(train_examples)
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fwci,
            generator=generator,
            drop_last=False,
        )
        total_loss = 0.0
        total_seen = 0
        progress = tqdm(loader, desc=f"train epoch {epoch:02d}/{epochs}", unit="batch")
        if len(loader) == 0:
            raise RuntimeError(
                f"epoch {epoch} has no training batches; "
                f"train_examples={len(train_examples)} batch_size={batch_size}"
            )
        for texts, y in progress:
            y = y.to(device)
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(inputs["input_ids"], inputs["attention_mask"])
                loss, _ = head_loss(pred, y, lambda_margin, lambda_rank)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * y.shape[0]
            total_seen += y.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")
        if skip_train_eval:
            train_metrics = {"rank_acc": 0.0, "corr_margin": 0.0}
        else:
            train_metrics = evaluate_fwci(
                model,
                tokenizer,
                train_examples,
                train_y,
                batch_size,
                max_length,
                lambda_margin,
                lambda_rank,
                "train",
            )
        val_metrics = evaluate_fwci(
            model,
            tokenizer,
            val_examples,
            val_y,
            batch_size,
            max_length,
            lambda_margin,
            lambda_rank,
            "val",
        )
        print(
            f"epoch={epoch:02d} batch_loss={total_loss / max(1, total_seen):.4f} "
            f"batches={len(loader)} "
            f"train_rank={train_metrics['rank_acc']:.4f} train_corr_m={train_metrics['corr_margin']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_rank={val_metrics['rank_acc']:.4f} "
            f"val_corr_m={val_metrics['corr_margin']:.4f} val_margin_mae={val_metrics['margin_mae']:.4f}"
        )
        metric = val_metrics["corr_margin"] + val_metrics["rank_acc"]
        if metric > best_metric:
            best_metric = metric
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
                "lora": {
                    k: v.detach().cpu().clone()
                    for k, v in model.backbone.state_dict().items()
                    if "lora_" in k
                },
            }
        if checkpoint_dir is not None:
            epoch_dir = checkpoint_dir / f"epoch_{epoch:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(epoch_dir / "lora")
            torch.save(
                {
                    "state_dict": model.head.state_dict(),
                    "hidden_size": model.head.in_features,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                epoch_dir / "head.pt",
            )
    if best_state is not None:
        model.head.load_state_dict(best_state["head"])
        backbone_state = model.backbone.state_dict()
        for key, value in best_state["lora"].items():
            if key in backbone_state:
                backbone_state[key] = value.to(backbone_state[key].device)
        model.backbone.load_state_dict(backbone_state, strict=False)

def load_model_from_epoch(
    device: torch.device,
    epoch_dir: Path,
    *,
    base_model: str | None = None,
) -> tuple[ComparatorRegressionModel, AutoTokenizer, dict]:
    base_path = base_model or base_model_path()
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    head_ckpt = torch.load(epoch_dir / "head.pt", map_location="cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=torch.cuda.is_available(),
    ).to(device)
    backbone = PeftModel.from_pretrained(
        base,
        epoch_dir / "lora",
        is_trainable=False,
    ).eval()
    hidden_size = int(head_ckpt.get("hidden_size", getattr(backbone.config, "hidden_size", 0)))
    model = ComparatorRegressionModel(backbone, hidden_size).to(device)
    model.head.load_state_dict(head_ckpt["state_dict"])
    model.eval()
    return model, tokenizer, head_ckpt

def save_comparator_weights(
    model: ComparatorRegressionModel,
    *,
    head_path: Path,
    lora_path: Path,
    target_mean: float,
    target_std: float,
    val_metrics: dict,
    test_metrics: dict,
    train_data: list[str],
    test_data: list[str],
    init_lora: str,
    lambda_margin: float,
    lambda_rank: float,
    swaps: bool,
) -> None:
    head_path.parent.mkdir(parents=True, exist_ok=True)
    lora_path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(lora_path)
    torch.save(
        {
            "state_dict": model.head.state_dict(),
            "hidden_size": model.head.in_features,
            "target": "normalized_log1p_fwci_pair",
            "target_mean": target_mean,
            "target_std": target_std,
            "loss": {
                "value": "smooth_l1(score_A/B, normalized_log1p_fwci_A/B)",
                "margin": "smooth_l1(score_A-score_B, target_A-target_B)",
                "rank": "weighted_bce_with_logits(score_A-score_B, fwci_A>fwci_B)",
                "lambda_margin": lambda_margin,
                "lambda_rank": lambda_rank,
            },
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "train_data": train_data,
            "test_data": test_data,
            "swaps": swaps,
            "lora_output": str(lora_path),
            "base_lora_path": init_lora,
            "training": "joint_lora_and_regression_head",
        },
        head_path,
    )
