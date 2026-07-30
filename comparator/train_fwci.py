from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SciNexus.comparator.model import (
    evaluate_fwci,
    load_fwci_examples,
    load_model_from_epoch,
    load_trainable_model,
    normalize_targets,
    save_comparator_weights,
    split_examples,
    train_fwci_model,
)
from SciNexus import paths as P
from SciNexus.comparator.paths import (
    base_model_path,
    default_checkpoint_dir,
    default_eval_output,
    default_head_output,
    default_lora_output,
    default_test_data,
    default_train_data,
    init_lora_path,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", nargs="+", default=[default_train_data()])
    parser.add_argument("--test-data", nargs="*", default=default_test_data())
    parser.add_argument("--output", default=default_head_output())
    parser.add_argument("--lora-output", default=default_lora_output())
    parser.add_argument("--init-lora", default=None, help="Initial LoRA checkpoint (or COMPARATOR_INIT_LORA)")
    parser.add_argument("--base-model", default=None, help="Qwen2.5-7B base model path")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lambda-margin", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.3)
    parser.add_argument("--no-swaps", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument("--checkpoint-dir", default=default_checkpoint_dir())
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-epoch", type=int, default=5)
    parser.add_argument("--eval-output", default=default_eval_output())
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output = Path(P.resolve_pkg_path_str(args.output))
    lora_output = Path(P.resolve_pkg_path_str(args.lora_output))
    checkpoint_dir = Path(P.resolve_pkg_path_str(args.checkpoint_dir))
    eval_output = Path(P.resolve_pkg_path_str(args.eval_output))
    output.parent.mkdir(parents=True, exist_ok=True)
    lora_output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    init_lora = args.init_lora or init_lora_path()
    base_model = args.base_model or base_model_path()

    if args.eval_only:
        epoch_dir = checkpoint_dir / f"epoch_{args.eval_epoch:02d}"
        if not (epoch_dir / "head.pt").exists():
            raise FileNotFoundError(f"missing checkpoint: {epoch_dir / 'head.pt'}")
        print(f"eval-only: loading epoch {args.eval_epoch:02d} from {epoch_dir} on {device}...")
        model, tokenizer, head_ckpt = load_model_from_epoch(
            device, epoch_dir, base_model=base_model
        )
        target_mean = float(head_ckpt["target_mean"])
        target_std = float(head_ckpt["target_std"])
    else:
        print(f"loading trainable comparator on {device}...")
        model, tokenizer = load_trainable_model(
            device,
            base_model=base_model,
            init_lora=init_lora,
        )
        target_mean = 0.0
        target_std = 1.0

    train_all = load_fwci_examples(
        [P.resolve_pkg_path(p) for p in args.train_data],
        tokenizer,
        add_swaps=not args.no_swaps,
    )
    if not train_all:
        raise RuntimeError("no training examples loaded; check --train-data and fwci fields")
    train_examples, val_examples = split_examples(train_all, args.val_frac, args.seed)
    test_sets = [
        (
            P.resolve_pkg_path(p).name,
            load_fwci_examples([P.resolve_pkg_path(p)], tokenizer, add_swaps=False),
        )
        for p in args.test_data
    ]
    print(
        f"train_total={len(train_all)} train={len(train_examples)} val={len(val_examples)} "
        f"swaps={not args.no_swaps}"
    )
    for name, examples in test_sets:
        print(f"test {name}: {len(examples)}")

    train_y_raw = torch.tensor([ex.target for ex in train_examples], dtype=torch.float32)
    val_y_raw = torch.tensor([ex.target for ex in val_examples], dtype=torch.float32)
    test_raw = [
        (name, torch.tensor([ex.target for ex in examples], dtype=torch.float32))
        for name, examples in test_sets
    ]
    train_y, normalized, target_mean, target_std = normalize_targets(
        train_y_raw,
        val_y_raw,
        *[y for _, y in test_raw],
    )
    val_y = normalized[0]
    test_norm = [
        (name, examples, y)
        for (name, examples), y in zip(test_sets, normalized[1:])
    ]
    if not args.eval_only:
        target_mean = float(target_mean if target_mean else 0.0)
        target_std = float(target_std if target_std else 1.0)
    print(f"target log1p(FWCI) mean={target_mean:.6f} std={target_std:.6f}")

    if not args.eval_only:
        print("joint fine-tuning LoRA + regression head...")
        train_fwci_model(
            model,
            tokenizer,
            train_examples,
            train_y.to(device),
            val_examples,
            val_y.to(device),
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            lora_lr=args.lora_lr,
            head_lr=args.head_lr,
            lambda_margin=args.lambda_margin,
            lambda_rank=args.lambda_rank,
            seed=args.seed,
            skip_train_eval=args.skip_train_eval,
            checkpoint_dir=checkpoint_dir,
            target_mean=target_mean,
            target_std=target_std,
        )

    val_metrics = evaluate_fwci(
        model,
        tokenizer,
        val_examples,
        val_y,
        args.batch_size,
        args.max_length,
        args.lambda_margin,
        args.lambda_rank,
        "val",
    )
    print(f"final val metrics: {val_metrics}")
    test_metrics = {}
    for name, examples, y in test_norm:
        metrics = evaluate_fwci(
            model,
            tokenizer,
            examples,
            y,
            args.batch_size,
            args.max_length,
            args.lambda_margin,
            args.lambda_rank,
            name,
        )
        test_metrics[name] = metrics
        print(f"final test metrics {name}: {metrics}")

    eval_payload = {
        "eval_epoch": args.eval_epoch if args.eval_only else args.epochs,
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    eval_output.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    print(f"saved eval summary {eval_output}")

    if args.eval_only:
        return

    save_comparator_weights(
        model,
        head_path=output,
        lora_path=lora_output,
        target_mean=target_mean,
        target_std=target_std,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_data=[str(p) for p in args.train_data],
        test_data=[str(p) for p in args.test_data],
        init_lora=init_lora,
        lambda_margin=args.lambda_margin,
        lambda_rank=args.lambda_rank,
        swaps=not args.no_swaps,
    )
    print(f"saved head checkpoint {output}")
    print(f"saved LoRA adapter {lora_output}")

if __name__ == "__main__":
    main()
