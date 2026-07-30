from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SciNexus import paths as P
from SciNexus.comparator.comparator import HypothesisComparator
from SciNexus.config import DEFAULT_CONFIG, resolve_config_paths, validate_external_paths

def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0:
        return None
    return out

def pearson(x: list[float], y: list[float]) -> float:
    if not x:
        return 0.0
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    xt = xt - xt.mean()
    yt = yt - yt.mean()
    denom = xt.norm() * yt.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((xt @ yt / denom).item())

def eval_file(comparator: HypothesisComparator, path: Path) -> dict:
    pred_margins: list[float] = []
    target_margins: list[float] = []
    rank_hits = 0
    total = 0
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for rec in tqdm(rows, desc=f"eval {path.name}", unit="pair"):
        hyp_a = rec.get("idea_A_json", "")
        hyp_b = rec.get("idea_B_json", "")
        fwci_a = _safe_float(rec.get("fwci_A"))
        fwci_b = _safe_float(rec.get("fwci_B"))
        if not hyp_a or not hyp_b or fwci_a is None or fwci_b is None:
            continue
        task = {
            "current_method": rec.get("current_method", ""),
            "limitation": rec.get("limitation", ""),
        }
        margin = comparator.compare_margin(task, hyp_a, hyp_b, use_cache=False)
        target = math.log1p(fwci_a) - math.log1p(fwci_b)
        pred_margins.append(margin)
        target_margins.append(target)
        rank_hits += int((margin > 0) == (target > 0))
        total += 1
    rank_acc = rank_hits / total if total else 0.0
    margin_mae = (
        sum(abs(p - t) for p, t in zip(pred_margins, target_margins)) / total
        if total
        else 0.0
    )
    return {
        "count": total,
        "rank_acc": rank_acc,
        "corr_margin": pearson(pred_margins, target_margins),
        "margin_mae": margin_mae,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-data",
        nargs="+",
        default=[P.COMPARATOR_TEST_DATA_2024, P.COMPARATOR_TEST_DATA_2025],
    )
    parser.add_argument("--head-path", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = resolve_config_paths(DEFAULT_CONFIG)
    if args.head_path:
        config.comparator_head_path = args.head_path
    if args.lora_path:
        config.comparator_lora_path = args.lora_path
    if args.base_model:
        config.comparator_base_model_path = args.base_model
    config = resolve_config_paths(config)
    validate_external_paths(config, require=("comparator_base_model_path",))

    print("loading HypothesisComparator...")
    comparator = HypothesisComparator(config)

    if not args.test_data:
        raise ValueError("Missing required path: --test-data")

    results = {}
    for p in args.test_data:
        path = P.resolve_pkg_path(p)
        if not path.exists():
            print(f"[WARN] skip missing {path}")
            continue
        metrics = eval_file(comparator, path)
        results[path.name] = metrics
        print(f"{path.name}: {metrics}")

    if args.output:
        out = P.resolve_pkg_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved {out}")

if __name__ == "__main__":
    main()
