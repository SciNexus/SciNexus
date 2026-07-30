from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "log"
RESULTS_DIR = "results"
TRAJECTORY_PATH = "pretrain_data/trajectories/pretrain_trajectories.jsonl"
PHASE1_SAVE_PATH = "checkpoints/phase1_evox.pt"
PHASE1_BEST_SAVE_PATH = "checkpoints/phase1_evox_best.pt"
PHASE2_INIT_CHECKPOINT = "checkpoints/phase1_evox_best.pt"
PHASE2_TASK_PATH = "pretrain_data/paper_datasets/data2pretrain.jsonl"
PHASE2_SAVE_PATH = "checkpoints/phase2_evox.pt"
PHASE2_TRACE_PATH = "log/phase2_evox_trace.jsonl"
PHASE2_LOSS_TRACE_PATH = "log/phase2_evox_trace_loss.jsonl"
PHASE2_TRAJECTORY_PATH = "log/phase2_trajectories.jsonl"
PHASE2_ACTION_TOPK_TRACE_PATH = "log/phase2_action_top3_trace.jsonl"
COMPARATOR_LORA_PATH = "checkpoints/comparator_reward_head_lora"
COMPARATOR_HEAD_PATH = "checkpoints/comparator_reward_head.pt"
COMPARATOR_TRAIN_DATA = "data/comparator_sample200/train.jsonl"
COMPARATOR_TEST_DATA_2024 = "data/comparator_sample200/test_2024.jsonl"
COMPARATOR_TEST_DATA_2025 = "data/comparator_sample200/test_2025.jsonl"
COMPARATOR_FWCI_CHECKPOINT_DIR = "checkpoints/fwci_reward_head"
COMPARATOR_FWCI_EVAL_OUTPUT = "checkpoints/fwci_reward_head/eval_test.json"
BENCH_INTERDISCIPLINARY = "bench/interdisciplinary_bench.jsonl"
PAPER_DATASETS_INPUT = "pretrain_data/paper_datasets/paper_datasets.jsonl"
PAPER_DATASETS_OUTPUT = "pretrain_data/paper_datasets/paper_datasets.jsonl"
DATA2PRETRAIN_PATH = "pretrain_data/paper_datasets/data2pretrain.jsonl"
DOI_TOPIC_DOMAINS_PATH = "pretrain_data/paper_datasets/doi_topic_domains.jsonl"
PRETRAIN_TRAJECTORIES_PATH = "pretrain_data/trajectories/pretrain_trajectories.jsonl"
PRETRAIN_TASKS_CACHE = "pretrain_data/trajectories/expanded_tasks.jsonl"
BASELINE_RESULTS_DIR = "baseline/results"
BASELINE_EVAL_OUTPUT_DIR = "baseline/results/evaluation"

ENV_SENTENCE_MODEL = "SENTENCE_MODEL_PATH"
ENV_QWEN_BASE_MODEL = "QWEN_BASE_MODEL_PATH"
ENV_COMPARATOR_INIT_LORA = "COMPARATOR_INIT_LORA"
ENV_TASK_DATA = "TASK_DATA_PATH"
ENV_PDF_RESULTS = "PDF_RESULTS_PATH"
ENV_PDF_DIR = "PDF_DIR"
ENV_TOPIC_DOMAINS_FILE = "TOPIC_DOMAINS_FILE"
ENV_CROSSENCODER_MODEL = "CROSSENCODER_MODEL_PATH"
ENV_EVAL_METADATA = "EVAL_METADATA_PATH"

def resolve_pkg_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PKG / p

def resolve_pkg_path_str(path: str | Path) -> str:
    return str(resolve_pkg_path(path))

def env_path(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()

def require_env_path(name: str, label: str | None = None) -> str:
    value = env_path(name)
    if not value:
        display = label or name
        raise ValueError(
            f"Missing required path: set environment variable {name} "
            f"({display}) or pass the corresponding CLI argument."
        )
    return value

def require_path(value: str | Path | None, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required path: {label}")
    return text
