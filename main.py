from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable
import numpy as np
import torch
_REPO = Path(__file__).resolve().parents[1]
_PKG = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from SciNexus.config import DEFAULT_CONFIG, ModelConfig, resolve_config_paths, validate_external_paths
from SciNexus import paths as P
from SciNexus.graph.graph_state import GraphState
from SciNexus.graph.evox_policy import EvoXGraphPolicy
from SciNexus.environment.evox_env import EvoXEnv
from SciNexus.training.phase1_pretrain import Phase1Trainer, TrajectoryDataset
from SciNexus.hypothesis_text import normalize_hypothesis_text
from SciNexus.training.phase2_rl import PPOTrainer
from SciNexus.comparator.comparator import HypothesisComparator

def build_embed_fn(config: ModelConfig) -> Callable[[str], np.ndarray]:
    validate_external_paths(config, require=("sentence_model_name",))
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.sentence_model_name)
        cache: dict[str, np.ndarray] = {}
        def embed(text) -> np.ndarray:
            key = normalize_hypothesis_text(text) or " "
            if key not in cache:
                cache[key] = model.encode(key, convert_to_numpy=True)
            return cache[key]
        print(f"[Embedder] {config.sentence_model_name}")
        return embed
    except ImportError:
        print("[Embedder] fallback zero vectors")
        def embed_fallback(text) -> np.ndarray:
            key = normalize_hypothesis_text(text) or " "
            rng = np.random.RandomState(abs(hash(key)) % (2**31))
            v = rng.randn(config.text_embed_dim).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-8)
        return embed_fallback
def build_role_embeddings(config: ModelConfig) -> torch.Tensor:
    gs = GraphState(
        agent_roles=config.agent_roles,
        interaction_types=config.interaction_types,
        text_embed_dim=config.text_embed_dim,
        role_embed_dim=config.role_embed_dim,
    )
    rows = [gs.get_role_embedding(i).numpy() for i in range(gs.num_nodes)]
    return torch.tensor(np.stack(rows), dtype=torch.float32)
def build_policy(config: ModelConfig) -> EvoXGraphPolicy:
    return EvoXGraphPolicy(
        num_nodes=len(config.agent_roles),
        num_relations=len(config.interaction_types),
        role_embed_dim=config.role_embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gnn_layers,
        task_embed_dim=config.text_embed_dim,
        dropout=config.gnn_dropout,
        use_relgnn=config.policy_use_relgnn,
        use_gru_memory=config.policy_use_gru_memory,
    )
def load_policy_weights(policy: EvoXGraphPolicy, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing:
        print(f"  [Policy] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [Policy] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"  Loaded policy weights from {ckpt_path}")
def cmd_pretrain(config: ModelConfig, embed_fn: Callable | None = None) -> EvoXGraphPolicy:
    print("\n═══ SciNexus Phase 1 — Behavioural Cloning ═══")
    validate_external_paths(config, require=("sentence_model_name",))
    embed_fn = embed_fn or build_embed_fn(config)
    records = Phase1Trainer.load_trajectories(config.trajectory_path)
    if not records:
        raise FileNotFoundError(f"No trajectories at {config.trajectory_path}")
    random.shuffle(records)
    split = int(0.9 * len(records))
    train_ds = TrajectoryDataset(records[:split], config, embed_fn)
    val_ds = TrajectoryDataset(records[split:], config, embed_fn) if split < len(records) else None
    policy = build_policy(config)
    role_emb = build_role_embeddings(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Phase1Trainer(policy, config, role_emb, device=device)
    trainer.train(train_ds, val_ds)
    return policy
def cmd_train_rl(
    config: ModelConfig,
    policy: EvoXGraphPolicy | None = None,
    embed_fn: Callable | None = None,
) -> EvoXGraphPolicy:
    print("\n═══ SciNexus Phase 2 — PPO Fine-tuning ═══")
    embed_fn = embed_fn or build_embed_fn(config)
    policy = policy or build_policy(config)
    init_ckpt = config.phase2_init_checkpoint
    if Path(init_ckpt).exists():
        load_policy_weights(policy, init_ckpt)
    elif Path(config.phase1_save_path).exists():
        load_policy_weights(policy, config.phase1_save_path)
    models = config.llm_models or [config.llm_model]
    workers = max(1, int(config.phase2_llm_workers))
    sync = max(1, int(config.phase2_sync_episodes))
    if len(models) > 1:
        slot_map = ", ".join(
            f"slot{i}→{models[i % len(models)]}" for i in range(min(workers, len(models)))
        )
        print(f"  LLM models: {models}  ({slot_map})")
    else:
        print(f"  LLM model: {models[0]}  (all {workers} workers)")
    print(f"  Parallel rollout: workers={workers}  sync_batch={sync}")
    validate_external_paths(
        config,
        require=("sentence_model_name", "comparator_base_model_path"),
    )
    comparator = HypothesisComparator(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    env = EvoXEnv(
        config=config,
        embed_fn=embed_fn,
        comparator_fn=comparator,
        policy=policy,
    )
    tasks = PPOTrainer.load_tasks(config.phase2_task_path)
    if not tasks:
        raise ValueError(f"No tasks loaded from {config.phase2_task_path}")
    if config.phase2_num_episodes <= 0:
        config.phase2_num_episodes = len(tasks)
    print(f"  Loaded {len(tasks)} tasks from {config.phase2_task_path}")
    print(f"  Episodes: {config.phase2_num_episodes}")
    trainer = PPOTrainer(
        policy,
        env,
        config,
        device=device,
        policy_factory=lambda: build_policy(config),
    )
    trainer.train(tasks)
    trainer.save()
    return policy
def cmd_run(
    config: ModelConfig,
    current_method: str,
    limitation: str,
    policy: EvoXGraphPolicy | None = None,
    embed_fn: Callable | None = None,
) -> dict:
    print("\n═══ SciNexus Inference ═══")
    embed_fn = embed_fn or build_embed_fn(config)
    policy = policy or build_policy(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    for ckpt_path in (config.phase2_save_path, config.phase1_save_path):
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(ckpt["policy"])
            print(f"  Loaded weights from {ckpt_path}")
            break
    env = EvoXEnv(config=config, embed_fn=embed_fn, policy=policy)
    task = {
        "current_method": current_method,
        "limitation": limitation,
    }
    obs = env.reset(task)
    policy.eval()
    last_action_idx: int | None = None
    with torch.no_grad():
        for step in range(config.max_steps):
            mem = (
                obs.memory.to(device)
                if obs.memory is not None
                else policy.init_memory(
                    obs.task_embedding.to(device),
                    env._role_emb_matrix.to(device),
                    device=device,
                )
            )
            sm, dm = env.get_action_masks()
            force_term = env.graph.current_step >= config.max_steps
            action, _, _ = policy.select_action(
                mem,
                obs.adj.to(device),
                obs.task_embedding.to(device),
                obs.active_mask.to(device),
                src_mask=sm.to(device),
                dst_mask=dm.to(device),
                deterministic=True,
                force_terminate=force_term,
                forbidden_flat_idx=last_action_idx,
                forbid_self_loop=config.policy_forbid_self_loop,
                forbid_repeat=config.policy_forbid_repeat_action,
            )
            out = policy(
                mem,
                obs.adj.to(device),
                obs.task_embedding.to(device),
                obs.active_mask.to(device),
                src_mask=sm.to(device),
                dst_mask=dm.to(device),
                forbidden_flat_idx=last_action_idx,
                forbid_self_loop=config.policy_forbid_self_loop,
                forbid_repeat=config.policy_forbid_repeat_action,
            )
            label = (
                "TERMINATE"
                if action["is_terminate"]
                else config.interaction_types[action["type"]]
            )
            dst = (
                ""
                if action["is_terminate"]
                else config.agent_roles[action["dst"]]
            )
            print(f"  Step {step + 1:2d} | {label:12s} → {dst}")
            obs = env.step(action, term_prob=out["term_prob"].item())
            if not action["is_terminate"]:
                last_action_idx = action["flat_idx"]
            if obs.done:
                break
    final_hyp = env._current_best_hyp or env.graph.get_best_hypothesis() or env._seed_hyp
    print(f"\n  Final hypothesis:\n  {final_hyp}\n")
    return {
        "task": task,
        "final_hypothesis": final_hyp,
        "episode_log": env.episode_log,
        "all_hypotheses": env.graph.get_all_hypotheses(),
    }
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SciNexus multi-agent discovery system")
    p.add_argument("command", choices=["pretrain", "train_rl", "run"])
    p.add_argument("--method", default="")
    p.add_argument("--limitation", default="")
    p.add_argument("--output", default=P.RESULTS_DIR + "/evox_output.json")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-ep", type=int, default=None)
    p.add_argument("--init-checkpoint", default=None, help="Phase 2 init checkpoint path")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last episode in phase2 loss trace (append logs)",
    )
    p.add_argument(
        "--start-ep",
        type=int,
        default=None,
        help="Explicit start episode (overrides --resume auto-detect)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Phase2 parallel LLM rollout workers (default: config.phase2_llm_workers)",
    )
    p.add_argument(
        "--sync-batch",
        type=int,
        default=None,
        help="Episodes per PPO update batch (default: config.phase2_sync_episodes)",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="Single LLM for all rollout workers (overrides config.llm_models)",
    )
    return p.parse_args()
def main() -> None:
    if len(sys.argv) > 1:
        if sys.argv[1] == "train_comparator":
            from SciNexus.comparator.train_fwci import main as train_comparator_main
            sys.argv = ["train_fwci"] + sys.argv[2:]
            train_comparator_main()
            return
        if sys.argv[1] == "eval_comparator":
            from SciNexus.comparator.eval import main as eval_comparator_main
            sys.argv = ["eval"] + sys.argv[2:]
            eval_comparator_main()
            return
        if sys.argv[1] == "train_comparator_head":
            from SciNexus.comparator.train_head import main as train_head_main
            sys.argv = ["train_head"] + sys.argv[2:]
            train_head_main()
            return
    args = parse_args()
    config = resolve_config_paths(DEFAULT_CONFIG)
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.num_ep:
        config.phase2_num_episodes = args.num_ep
    if args.init_checkpoint:
        ckpt = Path(args.init_checkpoint)
        config.phase2_init_checkpoint = str(P.resolve_pkg_path(ckpt) if not ckpt.is_absolute() else ckpt)
    elif args.resume:
        resume_ckpt = Path(config.phase2_save_path)
        if resume_ckpt.exists():
            config.phase2_init_checkpoint = str(resume_ckpt)
    if args.resume:
        config.phase2_resume = True
    if args.start_ep:
        config.phase2_start_episode = args.start_ep
    if args.workers:
        config.phase2_llm_workers = args.workers
    if args.sync_batch:
        config.phase2_sync_episodes = args.sync_batch
    if args.llm_model:
        config.llm_model = args.llm_model
        config.llm_models = [args.llm_model]
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    if args.command == "pretrain":
        cmd_pretrain(config)
    elif args.command == "train_rl":
        cmd_train_rl(config)
    elif args.command == "run":
        if not args.method or not args.limitation:
            raise ValueError("--method and --limitation are required for 'run'.")
        result = cmd_run(config, args.method, args.limitation)
        output_path = P.resolve_pkg_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Saved → {output_path}")
if __name__ == "__main__":
    main()
