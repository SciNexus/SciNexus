from __future__ import annotations

from SciNexus import paths as P

def base_model_path() -> str:
    return P.require_env_path(P.ENV_QWEN_BASE_MODEL, "Qwen2.5-7B base model")

def init_lora_path() -> str:
    return P.require_env_path(P.ENV_COMPARATOR_INIT_LORA, "initial LoRA checkpoint")

def default_train_data() -> str:
    return P.COMPARATOR_TRAIN_DATA

def default_test_data() -> list[str]:
    return [P.COMPARATOR_TEST_DATA_2024, P.COMPARATOR_TEST_DATA_2025]

def default_head_output() -> str:
    return P.COMPARATOR_HEAD_PATH

def default_lora_output() -> str:
    return P.COMPARATOR_LORA_PATH

def default_checkpoint_dir() -> str:
    return P.COMPARATOR_FWCI_CHECKPOINT_DIR

def default_eval_output() -> str:
    return P.COMPARATOR_FWCI_EVAL_OUTPUT

def resolve_pkg_path(path: str):
    return P.resolve_pkg_path(path)

def resolve_pkg_path_str(path: str) -> str:
    return P.resolve_pkg_path_str(path)
