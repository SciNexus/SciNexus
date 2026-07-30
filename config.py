from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from SciNexus import paths as P

@dataclass
class ModelConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    llm_model: str = "qwen3-235b-a22b"
    llm_models: List[str] = field(default_factory=lambda: ["qwen3-235b-a22b"])
    llm_temperature: float = 0.0
    llm_max_workers: int = 100
    sentence_model_name: str = field(default_factory=lambda: P.env_path(P.ENV_SENTENCE_MODEL))
    text_embed_dim: int = 384
    agent_roles: List[str] = field(default_factory=lambda: [
        "Life Scientist",
        "Chemist",
        "Computer Scientist",
        "Mathematician",
        "Physicist",
        "Earth Scientist",
        "Materials Scientist",
    ])
    interaction_types: List[str] = field(default_factory=lambda: [
        "debate",
        "analogy",
        "review",
        "clarification",
    ])
    role_embed_dim: int = 32
    hidden_dim: int = 256
    num_gnn_layers: int = 3
    gnn_dropout: float = 0.1
    max_steps: int = 10
    min_quality_delta: float = 0.02
    quality_window: int = 3
    reward_step_scale: float = 0.1
    reward_terminal_scale: float = 1.0
    lambda_local: float = 1.0
    lambda_global: float = 1.0
    lambda_step: float = 0.01
    step_tau: float = 2.0
    terminal_alpha: float = 0.5
    terminal_tau: float = 2.0
    use_step_reward: bool = True
    use_terminal_reward: bool = True
    policy_use_relgnn: bool = True
    policy_use_gru_memory: bool = True
    phase1_num_trajectories: int = 1038
    phase1_epochs: int = 10
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 20
    phase1_save_path: str = P.PHASE1_SAVE_PATH
    phase1_best_save_path: str = P.PHASE1_BEST_SAVE_PATH
    phase1_early_stop_patience: int = 3
    phase1_early_stop_min_delta: float = 1e-4
    phase2_num_episodes: int = 0
    phase2_lr: float = 3e-5
    phase2_gamma: float = 0.99
    phase2_gae_lambda: float = 0.95
    phase2_clip_eps: float = 0.2
    phase2_value_coef: float = 0.5
    phase2_entropy_coef: float = 0.01
    phase2_update_epochs: int = 4
    phase2_mini_batch_size: int = 16
    phase2_max_grad_norm: float = 0.5
    phase2_mix_alpha: float = 0.85
    phase2_llm_workers: int = 16
    phase2_sync_episodes: int = 16
    phase2_random_task_sampling: bool = False
    phase2_rollout_device: str = "cpu"
    phase2_sample_terminate: bool = True
    phase2_min_terminate_steps: int = 3
    phase2_terminate_explore_min_prob: float = 0.03
    phase2_terminate_explore_max_prob: float = 0.20
    phase2_action_temperature: float = 1.5
    phase2_repeat_action_penalty: float = 2.0
    phase2_self_loop_logit_penalty: float = 2.0
    policy_forbid_self_loop: bool = True
    policy_forbid_repeat_action: bool = True
    phase2_init_checkpoint: str = P.PHASE2_INIT_CHECKPOINT
    phase2_task_path: str = P.PHASE2_TASK_PATH
    phase2_save_path: str = P.PHASE2_SAVE_PATH
    phase2_trace_path: str = P.PHASE2_TRACE_PATH
    phase2_loss_trace_path: str = P.PHASE2_LOSS_TRACE_PATH
    phase2_trajectory_path: str = P.PHASE2_TRAJECTORY_PATH
    phase2_action_topk_trace_path: str = P.PHASE2_ACTION_TOPK_TRACE_PATH
    phase2_start_episode: int = 0
    phase2_resume: bool = False
    comparator_base_model_path: str = field(default_factory=lambda: P.env_path(P.ENV_QWEN_BASE_MODEL))
    comparator_lora_path: str = P.COMPARATOR_LORA_PATH
    comparator_head_path: str = P.COMPARATOR_HEAD_PATH
    comparator_head_temperature: float = 2.0
    comparator_head_position_swap: bool = True
    comparator_max_new_tokens: int = 512
    comparator_max_input_length: int = 4096
    task_data_path: str = field(default_factory=lambda: P.env_path(P.ENV_TASK_DATA))
    trajectory_path: str = P.TRAJECTORY_PATH
    checkpoint_dir: str = P.CHECKPOINT_DIR
    results_dir: str = P.RESULTS_DIR
    pdf_results_path: str = field(default_factory=lambda: P.env_path(P.ENV_PDF_RESULTS))
    pdf_dir: str = field(default_factory=lambda: P.env_path(P.ENV_PDF_DIR))
    topic_domains_file: str = field(default_factory=lambda: P.env_path(P.ENV_TOPIC_DOMAINS_FILE))
    crossencoder_model_path: str = field(default_factory=lambda: P.env_path(P.ENV_CROSSENCODER_MODEL))
    eval_metadata_path: str = field(default_factory=lambda: P.env_path(P.ENV_EVAL_METADATA))

DEFAULT_CONFIG = ModelConfig()

PKG_RELATIVE_PATH_FIELDS = (
    "trajectory_path",
    "phase1_save_path",
    "phase1_best_save_path",
    "phase2_init_checkpoint",
    "phase2_task_path",
    "phase2_save_path",
    "phase2_trace_path",
    "phase2_loss_trace_path",
    "phase2_trajectory_path",
    "phase2_action_topk_trace_path",
    "comparator_head_path",
    "comparator_lora_path",
    "checkpoint_dir",
    "results_dir",
)

EXTERNAL_PATH_FIELDS = (
    ("sentence_model_name", P.ENV_SENTENCE_MODEL, "sentence embedding model"),
    ("comparator_base_model_path", P.ENV_QWEN_BASE_MODEL, "Qwen2.5-7B base model"),
    ("task_data_path", P.ENV_TASK_DATA, "task dataset jsonl"),
    ("pdf_results_path", P.ENV_PDF_RESULTS, "pdf results jsonl"),
    ("pdf_dir", P.ENV_PDF_DIR, "source PDF directory"),
    ("topic_domains_file", P.ENV_TOPIC_DOMAINS_FILE, "topic domains metadata jsonl"),
    ("crossencoder_model_path", P.ENV_CROSSENCODER_MODEL, "Cross-Encoder model directory"),
    ("eval_metadata_path", P.ENV_EVAL_METADATA, "evaluation metadata jsonl"),
)

def resolve_config_paths(config: ModelConfig) -> ModelConfig:
    for attr in PKG_RELATIVE_PATH_FIELDS:
        val = getattr(config, attr, None)
        if val and not os.path.isabs(str(val)):
            setattr(config, attr, P.resolve_pkg_path_str(val))
    return config

def validate_external_paths(
    config: ModelConfig,
    *,
    require: tuple[str, ...] = (),
) -> None:
    missing: list[str] = []
    for attr, env_name, label in EXTERNAL_PATH_FIELDS:
        if require and attr not in require:
            continue
        val = str(getattr(config, attr, "") or "").strip()
        if not val:
            missing.append(f"  {attr}: set {env_name} or pass --{attr.replace('_', '-')}")
    if missing:
        raise ValueError("Missing required external paths:\n" + "\n".join(missing))
