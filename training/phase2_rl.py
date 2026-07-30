from __future__ import annotations
import json
import random
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from SciNexus.config import ModelConfig
from ..environment.evox_env import EvoXEnv
from ..graph.evox_policy import (
    EvoXGraphPolicy,
    action_log_prob,
    interaction_entropy,
    pick_step_action_index,
)
from ..pretrain_data.task_utils import load_pretrain_tasks
@dataclass
class Transition:
    memory: torch.Tensor
    adj: torch.Tensor
    task_embedding: torch.Tensor
    active_mask: torch.Tensor
    src_mask: torch.Tensor
    dst_mask: torch.Tensor
    action_flat_idx: int
    step_count: int
    repeat_penalized_action_idx: int | None
    log_prob: float
    value: float
    mix_alpha: float
    stop_prob: float
    sampled_from_full: bool
    reward: float
    step_reward: float
    terminal_reward: float
    done: bool
@dataclass
class Rollout:
    transitions: list[Transition] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    episode_log: list[dict] = field(default_factory=list)
    action_topk_trace: list[dict] = field(default_factory=list)
    seed_hypothesis: str = ""
    final_hypothesis: str = ""
@dataclass
class CollectedRollout:
    episode_idx: int
    task: dict
    rollout: Rollout
    llm_model: str
def compute_gae(rollout: Rollout, gamma: float, lam: float) -> tuple[list[float], list[float]]:
    t_len = len(rollout.transitions)
    advantages = [0.0] * t_len
    returns = [0.0] * t_len
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(t_len)):
        tr = rollout.transitions[t]
        delta = tr.reward + gamma * next_value * (1 - int(tr.done)) - tr.value
        gae = delta + gamma * lam * (1 - int(tr.done)) * gae
        advantages[t] = gae
        returns[t] = gae + tr.value
        next_value = tr.value
    rollout.advantages = advantages
    rollout.returns = returns
    return advantages, returns
class PPOTrainer:
    def __init__(
        self,
        policy: EvoXGraphPolicy,
        env: EvoXEnv,
        config: ModelConfig,
        device: str = "cpu",
        policy_factory: Callable[[], EvoXGraphPolicy] | None = None,
    ):
        self.policy = policy.to(device)
        self.env = env
        self.config = config
        self.device = device
        self.policy_factory = policy_factory
        self.rollout_device = config.phase2_rollout_device
        self.num_workers = max(1, int(config.phase2_llm_workers))
        self.sync_episodes = max(1, int(config.phase2_sync_episodes))
        self._comparator_lock = threading.Lock()
        self._trajectory_lock = threading.Lock()
        self.optim = optim.Adam(policy.parameters(), lr=config.phase2_lr, eps=1e-5)
        self.mse = nn.MSELoss()
        self._episode_rewards: list[float] = []
    def train(self, tasks: list[dict], save_every: int = 5) -> list[float]:
        all_rewards: list[float] = []
        Path(self.config.phase2_trace_path).parent.mkdir(parents=True, exist_ok=True)
        start_ep = self._resolve_start_episode()
        resume = start_ep > 1
        if resume:
            self._load_resume_state()
            print(f"  [Phase2] Resuming from episode {start_ep}")
        else:
            with open(self.config.phase2_trace_path, "w", encoding="utf-8"):
                pass
            self._init_loss_trace_file()
            self._init_trajectory_file()
            self._init_action_topk_trace_file()
        total_episodes = self.config.phase2_num_episodes
        if start_ep > total_episodes:
            print(f"  [Phase2] Already completed ({start_ep - 1}/{total_episodes} episodes)")
            return self._episode_rewards
        sampling = (
            "random-unique/sync"
            if self.config.phase2_random_task_sampling
            else "sequential"
        )
        print(
            f"  [Phase2] workers={self.num_workers}  "
            f"sync_batch={self.sync_episodes}  rollout_device={self.rollout_device}  "
            f"task_sampling={sampling}  llm_temp={self.config.llm_temperature}  "
            f"episodes={start_ep}-{total_episodes}"
        )
        if self.num_workers <= 1:
            return self._train_sequential(tasks, all_rewards, save_every, start_ep)
        return self._train_parallel(tasks, all_rewards, save_every, total_episodes, start_ep)
    def _resolve_start_episode(self) -> int:
        if self.config.phase2_start_episode > 0:
            return self.config.phase2_start_episode
        if not self.config.phase2_resume:
            return 1
        return self.detect_last_episode(self.config.phase2_loss_trace_path) + 1
    @staticmethod
    def detect_last_episode(trace_path: str) -> int:
        path = Path(trace_path)
        if not path.exists() or path.stat().st_size == 0:
            return 0
        last_ep = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last_ep = max(last_ep, int(json.loads(line).get("episode", 0)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return last_ep
    def _load_resume_state(self) -> None:
        ckpt_path = Path(self.config.phase2_init_checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            prev = ckpt.get("rewards", [])
            if prev:
                self._episode_rewards = list(prev)
                print(f"  [Phase2] Restored {len(prev)} prior episode rewards from checkpoint")
        trace = Path(self.config.phase2_loss_trace_path)
        if trace.exists():
            print(f"  [Phase2] Appending to loss log → {trace}")
        traj = Path(self.config.phase2_trajectory_path)
        if traj.exists():
            print(f"  [Phase2] Appending to trajectory log → {traj}")
    def _sample_batch_tasks(self, tasks: list[dict], batch_size: int) -> list[dict]:
        if not tasks:
            raise ValueError("No tasks available for Phase2 sampling")
        k = min(batch_size, len(tasks))
        if self.config.phase2_random_task_sampling:
            return random.sample(tasks, k)
        start = getattr(self, "_next_task_idx", 0)
        batch = [tasks[(start + i) % len(tasks)] for i in range(k)]
        self._next_task_idx = start + k
        return batch
    def _train_sequential(
        self,
        tasks: list[dict],
        all_rewards: list[float],
        save_every: int,
        start_ep: int,
    ) -> list[float]:
        pbar = tqdm(
            range(start_ep, self.config.phase2_num_episodes + 1),
            desc="Phase2 RL",
            unit="ep",
            file=sys.stderr,
            initial=start_ep - 1,
            total=self.config.phase2_num_episodes,
        )
        self._next_task_idx = start_ep - 1
        for ep in pbar:
            task = self._sample_batch_tasks(tasks, 1)[0]
            llm_model = self._episode_llm_model(ep)
            self._set_llm_model(llm_model)
            rollout = self._collect_rollout(self.env, self.policy, task)
            self._finalize_episode(
                ep=ep,
                task=task,
                rollout=rollout,
                llm_model=llm_model,
                all_rewards=all_rewards,
                pg=None,
                val=None,
                ent=None,
            )
            if ep % save_every == 0:
                self.save()
            pbar.set_postfix(
                llm=llm_model,
                R=f"{all_rewards[-1]:+.3f}",
                refresh=False,
            )
        pbar.close()
        return all_rewards
    def _train_parallel(
        self,
        tasks: list[dict],
        all_rewards: list[float],
        save_every: int,
        total_episodes: int,
        start_ep: int,
    ) -> list[float]:
        next_episode_idx = start_ep
        update_idx = (start_ep - 1) // self.sync_episodes
        if not self.config.phase2_random_task_sampling:
            self._next_task_idx = start_ep - 1
        pbar = tqdm(
            total=total_episodes,
            desc="Phase2 RL",
            unit="ep",
            file=sys.stderr,
            initial=start_ep - 1,
        )
        with ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="evox-phase2",
        ) as executor:
            while next_episode_idx <= total_episodes:
                remaining = total_episodes - next_episode_idx + 1
                batch_size = min(self.sync_episodes, self.num_workers, remaining)
                batch_tasks = self._sample_batch_tasks(tasks, batch_size)
                batch_size = len(batch_tasks)
                policy_state = self._snapshot_policy_state()
                batch_start_ep = next_episode_idx
                in_flight: dict[Future, int] = {}
                for slot in range(batch_size):
                    ep = next_episode_idx
                    task = batch_tasks[slot]
                    llm_model = self._episode_llm_model(ep, slot=slot)
                    future = executor.submit(
                        self._worker_collect_rollout,
                        slot,
                        ep,
                        task,
                        llm_model,
                        policy_state,
                    )
                    in_flight[future] = slot
                    next_episode_idx += 1
                collected: list[CollectedRollout] = []
                for future in as_completed(in_flight):
                    collected.append(future.result())
                update_idx += 1
                ordered = sorted(collected, key=lambda item: item.episode_idx)
                for item in ordered:
                    compute_gae(
                        item.rollout,
                        self.config.phase2_gamma,
                        self.config.phase2_gae_lambda,
                    )
                pg, val, ent = self._ppo_update_batch([item.rollout for item in ordered])
                total_loss = (
                    pg
                    + self.config.phase2_value_coef * val
                    - self.config.phase2_entropy_coef * ent
                )
                for item in ordered:
                    step_r, term_r, ep_reward = self._reward_breakdown(item.rollout)
                    self._append_loss_trace(
                        ep=item.episode_idx,
                        llm_model=item.llm_model,
                        pg_loss=pg,
                        val_loss=val,
                        entropy=ent,
                        total_loss=total_loss,
                        ep_reward=ep_reward,
                        step_reward=step_r,
                        terminal_reward=term_r,
                        num_steps=len(item.rollout.transitions),
                    )
                    self._append_trajectory(
                        ep=item.episode_idx,
                        task=item.task,
                        rollout=item.rollout,
                        llm_model=item.llm_model,
                        ep_reward=ep_reward,
                    )
                    self._append_action_topk_trace(
                        ep=item.episode_idx,
                        task=item.task,
                        rollout=item.rollout,
                        llm_model=item.llm_model,
                    )
                    self._episode_rewards.append(ep_reward)
                    all_rewards.append(ep_reward)
                last_ep = ordered[-1].episode_idx
                if last_ep % save_every == 0:
                    self.save()
                avg_r = sum(all_rewards[-10:]) / min(10, len(all_rewards))
                pbar.update(len(ordered))
                pbar.set_postfix(
                    upd=update_idx,
                    batch=batch_size,
                    R=f"{all_rewards[-1]:+.3f}",
                    avg10=f"{avg_r:+.3f}",
                    loss=f"{total_loss:.4f}",
                    refresh=False,
                )
                if last_ep % 10 == 0 or batch_start_ep == start_ep:
                    batch_llms = ", ".join(
                        f"ep{item.episode_idx}:{item.llm_model}" for item in ordered
                    )
                    tqdm.write(
                        f"  [Phase2] ep {batch_start_ep}-{last_ep}/{total_episodes} "
                        f"upd={update_idx} batch={batch_size} "
                        f"pg={pg:.4f} v={val:.4f} ent={ent:.4f} "
                        f"[{batch_llms}]"
                    )
        pbar.close()
        return all_rewards
    def _finalize_episode(
        self,
        *,
        ep: int,
        task: dict,
        rollout: Rollout,
        llm_model: str,
        all_rewards: list[float],
        pg: float | None,
        val: float | None,
        ent: float | None,
    ) -> None:
        compute_gae(rollout, self.config.phase2_gamma, self.config.phase2_gae_lambda)
        step_r, term_r, ep_reward = self._reward_breakdown(rollout)
        self._episode_rewards.append(ep_reward)
        if pg is None:
            pg, val, ent = self._ppo_update(rollout)
        total_loss = (
            pg
            + self.config.phase2_value_coef * val
            - self.config.phase2_entropy_coef * ent
        )
        self._append_loss_trace(
            ep=ep,
            llm_model=llm_model,
            pg_loss=pg,
            val_loss=val,
            entropy=ent,
            total_loss=total_loss,
            ep_reward=ep_reward,
            step_reward=step_r,
            terminal_reward=term_r,
            num_steps=len(rollout.transitions),
        )
        self._append_trajectory(
            ep=ep,
            task=task,
            rollout=rollout,
            llm_model=llm_model,
            ep_reward=ep_reward,
        )
        self._append_action_topk_trace(
            ep=ep,
            task=task,
            rollout=rollout,
            llm_model=llm_model,
        )
        all_rewards.append(ep_reward)
    def _worker_collect_rollout(
        self,
        slot: int,
        episode_idx: int,
        task: dict,
        llm_model: str,
        policy_state: dict[str, torch.Tensor],
    ) -> CollectedRollout:
        worker_config = replace(self.config, llm_model=llm_model)
        local_policy = self._build_worker_policy(policy_state)
        env = self._build_worker_env(local_policy, worker_config)
        rollout = self._collect_rollout(env, local_policy, task)
        return CollectedRollout(
            episode_idx=episode_idx,
            task=task,
            rollout=rollout,
            llm_model=llm_model,
        )
    def _build_worker_policy(self, policy_state: dict[str, torch.Tensor]) -> EvoXGraphPolicy:
        if self.policy_factory is not None:
            worker = self.policy_factory()
        else:
            worker = deepcopy(self.policy)
        worker.load_state_dict(policy_state)
        worker.to(self.rollout_device)
        worker.eval()
        return worker
    def _build_worker_env(
        self,
        local_policy: EvoXGraphPolicy,
        worker_config: ModelConfig,
    ) -> EvoXEnv:
        return EvoXEnv(
            config=worker_config,
            embed_fn=self.env.embed_fn,
            comparator_fn=self._locked_comparator(),
            policy=local_policy,
        )
    def _locked_comparator(self) -> Any | None:
        comp = self.env.comparator
        if comp is None:
            return None
        lock = self._comparator_lock
        class _LockedComparator:
            def __call__(self, task: dict, hyp_a: str, hyp_b: str) -> int:
                with lock:
                    return comp(task, hyp_a, hyp_b)
            def compare_logits(self, task: dict, hyp_a: str, hyp_b: str):
                with lock:
                    return comp.compare_logits(task, hyp_a, hyp_b)
            def compare_scores(
                self,
                task: dict,
                hyp_a: str,
                hyp_b: str,
                use_cache: bool = True,
                position_swap: bool | None = None,
            ):
                with lock:
                    return comp.compare_scores(
                        task, hyp_a, hyp_b, use_cache=use_cache, position_swap=position_swap
                    )
            def compare_margin(
                self,
                task: dict,
                hyp_a: str,
                hyp_b: str,
                use_cache: bool = True,
                position_swap: bool | None = None,
            ) -> float:
                with lock:
                    return comp.compare_margin(
                        task, hyp_a, hyp_b, use_cache=use_cache, position_swap=position_swap
                    )
        return _LockedComparator()
    def _snapshot_policy_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}
    def _episode_llm_model(self, ep: int, slot: int | None = None) -> str:
        models = self.config.llm_models or [self.config.llm_model]
        if slot is not None and len(models) > 1:
            return models[slot % len(models)]
        return models[(ep - 1) % len(models)]
    def _set_llm_model(self, model: str) -> None:
        self.config.llm_model = model
    def _init_trajectory_file(self) -> None:
        path = Path(self.config.phase2_trajectory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        print(f"  [Phase2] Trajectory log → {path}")
    def _init_action_topk_trace_file(self) -> None:
        path = Path(self.config.phase2_action_topk_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        print(f"  [Phase2] Action top3 log → {path}")
    def _build_trajectory_steps(self, episode_log: list[dict]) -> list[dict]:
        skip = {"reset", "terminal_bonus"}
        steps: list[dict] = []
        for entry in episode_log:
            event = entry.get("event", "step")
            if event in skip:
                continue
            src = entry.get("src", "")
            dst = entry.get("dst", "")
            itype = entry.get("type", "")
            steps.append({
                "step": entry.get("step", len(steps)),
                "event": event,
                "src_role": src if src != "—" else "",
                "dst_role": dst if dst != "—" else "",
                "interaction_type": itype if itype not in {"—", "terminate"} else "",
                "type": itype if itype not in {"—", "terminate"} else "",
                "src_message": entry.get("src_message", ""),
                "hypothesis": entry.get("hypothesis", entry.get("hypothesis_snippet", "")),
                "reward": entry.get("reward", 0.0),
                "local_score_a": entry.get("local_score_a"),
                "local_score_b": entry.get("local_score_b"),
                "local_margin": entry.get("local_margin"),
                "global_score_a": entry.get("global_score_a"),
                "global_score_b": entry.get("global_score_b"),
                "global_margin": entry.get("global_margin"),
                "terminal_base_margin": entry.get("terminal_base_margin"),
                "terminal_best_margin": entry.get("terminal_best_margin"),
                "is_terminate": event in {"terminate_action", "max_steps"},
            })
        return steps
    def _describe_policy_actions(self, rollout: Rollout) -> list[dict]:
        n = self.policy.num_nodes
        k = self.policy.num_relations
        term_idx = n * n * k
        roles = self.config.agent_roles
        types = self.config.interaction_types
        actions: list[dict] = []
        for tr in rollout.transitions:
            idx = tr.action_flat_idx
            if idx == term_idx:
                action = {
                    "action": "TERMINATE",
                    "src": "",
                    "dst": "",
                    "interaction_type": "",
                    "flat_idx": idx,
                }
            else:
                rel = idx % k
                rem = idx // k
                dst = rem % n
                src = rem // n
                action = {
                    "action": types[rel],
                    "src": roles[src],
                    "dst": roles[dst],
                    "interaction_type": types[rel],
                    "flat_idx": idx,
                }
            action.update({
                "log_prob": tr.log_prob,
                "value": tr.value,
                "reward": tr.reward,
                "step_reward": tr.step_reward,
                "terminal_reward": tr.terminal_reward,
                "done": tr.done,
                "mix_alpha": tr.mix_alpha,
            })
            actions.append(action)
        return actions
    def _append_trajectory(
        self,
        *,
        ep: int,
        task: dict,
        rollout: Rollout,
        llm_model: str,
        ep_reward: float,
    ) -> None:
        step_r, term_r, ep_reward = self._reward_breakdown(rollout)
        record = {
            "task": {
                "current_method": task.get("current_method", ""),
                "limitation": task.get("limitation", ""),
            },
            "trajectory": self._build_trajectory_steps(rollout.episode_log),
            "policy_actions": self._describe_policy_actions(rollout),
            "meta": {
                "episode": ep,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "llm_model": llm_model,
                "total_reward": ep_reward,
                "step_reward": step_r,
                "terminal_reward": term_r,
                "num_steps": len(rollout.transitions),
                "seed_hypothesis": rollout.seed_hypothesis,
                "final_hypothesis": rollout.final_hypothesis,
                "source_file": task.get("source_file", ""),
                "limitation_idx": task.get("limitation_idx", 0),
            },
        }
        path = self.config.phase2_trajectory_path
        with self._trajectory_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    def _append_action_topk_trace(
        self,
        *,
        ep: int,
        task: dict,
        rollout: Rollout,
        llm_model: str,
    ) -> None:
        record = {
            "episode": ep,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "llm_model": llm_model,
            "task": {
                "current_method": task.get("current_method", ""),
                "limitation": task.get("limitation", ""),
            },
            "num_steps": len(rollout.transitions),
            "actions": rollout.action_topk_trace,
        }
        path = Path(self.config.phase2_action_topk_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._trajectory_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    def _init_loss_trace_file(self) -> None:
        path = Path(self.config.phase2_loss_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        print(f"  [Phase2] Loss log  → {path}")
    def _reward_breakdown(self, rollout: Rollout) -> tuple[float, float, float]:
        step_reward = sum(tr.step_reward for tr in rollout.transitions)
        terminal_reward = sum(tr.terminal_reward for tr in rollout.transitions)
        total_reward = sum(tr.reward for tr in rollout.transitions)
        return step_reward, terminal_reward, total_reward
    def _append_loss_trace(
        self,
        *,
        ep: int,
        llm_model: str,
        pg_loss: float,
        val_loss: float,
        entropy: float,
        total_loss: float,
        ep_reward: float,
        step_reward: float,
        terminal_reward: float,
        num_steps: int,
    ) -> None:
        record = {
            "episode": ep,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "llm_model": llm_model,
            "policy_loss": pg_loss,
            "value_loss": val_loss,
            "entropy": entropy,
            "total_loss": total_loss,
            "total_reward": ep_reward,
            "step_reward": step_reward,
            "terminal_reward": terminal_reward,
            "num_steps": num_steps,
        }
        with open(self.config.phase2_loss_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    def _mix_alpha(self) -> float:
        return max(0.0, min(1.0, float(self.config.phase2_mix_alpha)))
    def _mixed_probs(self, logits: torch.Tensor, alpha: float) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        temperature = max(1e-6, float(self.config.phase2_action_temperature))
        policy = torch.softmax(logits / temperature, dim=0)
        valid = (logits > -1e8).float()
        valid = valid / valid.sum().clamp_min(1.0)
        mixed = alpha * policy + (1.0 - alpha) * valid
        return mixed / mixed.sum().clamp_min(1e-12)
    def _scheduled_stop_prob(self, step_count: int) -> float:
        if not self.config.phase2_sample_terminate:
            return 0.0
        min_step = int(self.config.phase2_min_terminate_steps)
        if step_count < min_step:
            return 0.0
        max_steps = max(min_step + 1, int(self.config.max_steps))
        span = max(1, max_steps - min_step)
        progress = max(0.0, min(1.0, float(step_count - min_step) / float(span)))
        p_min = max(0.0, min(1.0, float(self.config.phase2_terminate_explore_min_prob)))
        p_max = max(p_min, min(1.0, float(self.config.phase2_terminate_explore_max_prob)))
        return p_min + (p_max - p_min) * progress
    def _terminate_sampling_probs(
        self,
        probs: torch.Tensor,
        step_count: int,
        term_idx: int,
    ) -> tuple[torch.Tensor, float]:
        stop_prob = self._scheduled_stop_prob(step_count)
        if stop_prob <= 0.0:
            return probs, 0.0
        stop_prob = max(stop_prob, float(probs[term_idx].detach().cpu()))
        stop_prob = max(0.0, min(1.0, stop_prob))
        interact = probs[:term_idx].clamp(min=1e-12)
        interact = interact / interact.sum().clamp(min=1e-12)
        adjusted = probs.clone()
        adjusted[:term_idx] = interact * (1.0 - stop_prob)
        adjusted[term_idx] = stop_prob
        return adjusted / adjusted.sum().clamp(min=1e-12), stop_prob
    def _decode_action(self, idx: int, n: int, k: int) -> dict:
        term_idx = n * n * k
        if idx == term_idx:
            return {
                "action": "TERMINATE",
                "src": "",
                "dst": "",
                "interaction_type": "",
                "flat_idx": idx,
            }
        rel = idx % k
        rem = idx // k
        dst = rem % n
        src = rem // n
        return {
            "action": self.config.interaction_types[rel],
            "src": self.config.agent_roles[src],
            "dst": self.config.agent_roles[dst],
            "interaction_type": self.config.interaction_types[rel],
            "flat_idx": idx,
        }
    def _topk_action_trace(
        self,
        *,
        step_count: int,
        probs: torch.Tensor,
        raw_term_prob: float,
        stop_prob: float,
        selected_idx: int,
        log_prob: float,
        sampled_from_full: bool,
        repeated_action_idx: int | None,
        n: int,
        k: int,
    ) -> dict:
        topv, topi = torch.topk(probs.detach().cpu(), k=min(3, probs.numel()))
        return {
            "step": step_count,
            "raw_term_prob": raw_term_prob,
            "scheduled_stop_prob": stop_prob,
            "sampled_from_full": sampled_from_full,
            "selected": self._decode_action(selected_idx, n, k),
            "selected_log_prob": log_prob,
            "repeat_penalized_flat_idx": repeated_action_idx,
            "top3": [
                {
                    **self._decode_action(int(idx), n, k),
                    "prob": float(prob),
                }
                for prob, idx in zip(topv.tolist(), topi.tolist())
            ],
        }
    def _collect_rollout(
        self,
        env: EvoXEnv,
        policy: EvoXGraphPolicy,
        task: dict,
    ) -> Rollout:
        rollout = Rollout()
        obs = env.reset(task)
        alpha = self._mix_alpha()
        rollout_dev = next(policy.parameters()).device
        role_emb = env._role_emb_matrix.to(rollout_dev)
        last_action_idx: int | None = None
        for _ in range(self.config.max_steps):
            mem = (
                obs.memory.to(rollout_dev)
                if obs.memory is not None
                else policy.init_memory(
                    obs.task_embedding.to(rollout_dev),
                    role_emb,
                    device=rollout_dev,
                )
            )
            adj = obs.adj.to(rollout_dev)
            te = obs.task_embedding.to(rollout_dev)
            am = obs.active_mask.to(rollout_dev)
            sm, dm = env.get_action_masks()
            sm, dm = sm.to(rollout_dev), dm.to(rollout_dev)
            with torch.no_grad():
                out = policy(
                    mem,
                    adj,
                    te,
                    am,
                    src_mask=sm,
                    dst_mask=dm,
                    forbidden_flat_idx=last_action_idx,
                    forbid_self_loop=self.config.policy_forbid_self_loop,
                    forbid_repeat=self.config.policy_forbid_repeat_action,
                )
                n, k = policy.num_nodes, policy.num_relations
                term_idx = n * n * k
                probs = self._mixed_probs(out["all_logits"], alpha)
                force_term = env.graph.current_step >= self.config.max_steps
                allow_term = env.graph.current_step >= self.config.phase2_min_terminate_steps
                sample_term = bool(self.config.phase2_sample_terminate and allow_term)
                if sample_term:
                    probs, stop_prob = self._terminate_sampling_probs(
                        probs,
                        env.graph.current_step,
                        term_idx,
                    )
                else:
                    stop_prob = 0.0
                idx, log_prob = pick_step_action_index(
                    probs,
                    policy.num_nodes,
                    policy.num_relations,
                    force_terminate=force_term,
                    sample_interaction=True,
                    allow_terminate=allow_term,
                    sample_terminate=sample_term,
                )
                value = out["value"]
                term_prob = out["term_prob"].item()
                rollout.action_topk_trace.append(
                    self._topk_action_trace(
                        step_count=env.graph.current_step,
                        probs=probs,
                        raw_term_prob=term_prob,
                        stop_prob=stop_prob,
                        selected_idx=idx,
                        log_prob=log_prob.item(),
                        sampled_from_full=sample_term,
                        repeated_action_idx=last_action_idx,
                        n=n,
                        k=k,
                    )
                )
            action = self._idx_to_action(idx, policy)
            next_obs = env.step(action, term_prob=term_prob)
            rollout.transitions.append(Transition(
                memory=mem.cpu(),
                adj=obs.adj,
                task_embedding=obs.task_embedding,
                active_mask=obs.active_mask,
                src_mask=sm.cpu(),
                dst_mask=dm.cpu(),
                action_flat_idx=idx,
                step_count=int(rollout.action_topk_trace[-1]["step"]),
                repeat_penalized_action_idx=last_action_idx,
                log_prob=log_prob.item(),
                value=value.item(),
                mix_alpha=alpha,
                stop_prob=stop_prob,
                sampled_from_full=sample_term,
                reward=next_obs.reward,
                step_reward=float(next_obs.info.get("step_reward", next_obs.reward)),
                terminal_reward=float(next_obs.info.get("terminal_reward", 0.0)),
                done=next_obs.done,
            ))
            last_action_idx = idx
            obs = next_obs
            if next_obs.done:
                break
        rollout.episode_log = list(env.episode_log)
        rollout.seed_hypothesis = env._seed_hyp
        rollout.final_hypothesis = (
            env._last_hypothesis
            or env._current_best_hyp
            or env.graph.get_best_hypothesis()
            or env._seed_hyp
        )
        return rollout
    def _idx_to_action(self, idx: int, policy: EvoXGraphPolicy) -> dict:
        n, k = policy.num_nodes, policy.num_relations
        term_idx = n * n * k
        if idx == term_idx:
            return {"src": -1, "dst": -1, "type": -1, "is_terminate": True, "flat_idx": idx}
        rel = idx % k
        rem = idx // k
        dst = rem % n
        src = rem // n
        return {"src": src, "dst": dst, "type": rel, "is_terminate": False, "flat_idx": idx}
    def _ppo_update(self, rollout: Rollout) -> tuple[float, float, float]:
        return self._ppo_update_batch([rollout])
    def _ppo_update_batch(self, rollouts: list[Rollout]) -> tuple[float, float, float]:
        transitions = [tr for rollout in rollouts for tr in rollout.transitions]
        if not transitions:
            return 0.0, 0.0, 0.0
        was_training = self.policy.training
        self.policy.train()
        adv = torch.tensor(
            [a for rollout in rollouts for a in rollout.advantages],
            dtype=torch.float32,
        )
        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False)
        adv = adv - adv_mean if adv_std < 1e-8 else (adv - adv_mean) / (adv_std + 1e-8)
        rets = torch.tensor(
            [ret for rollout in rollouts for ret in rollout.returns],
            dtype=torch.float32,
        )
        old_lps = torch.tensor([tr.log_prob for tr in transitions], dtype=torch.float32)
        total_pg = total_val = total_ent = 0.0
        n_updates = 0
        total_len = len(transitions)
        for _ in range(self.config.phase2_update_epochs):
            indices = list(range(total_len))
            random.shuffle(indices)
            for start in range(0, total_len, self.config.phase2_mini_batch_size):
                mb = indices[start : start + self.config.phase2_mini_batch_size]
                if not mb:
                    continue
                pg_loss = torch.tensor(0.0, device=self.device)
                val_loss = torch.tensor(0.0, device=self.device)
                entropy = torch.tensor(0.0, device=self.device)
                for i in mb:
                    tr = transitions[i]
                    mem = tr.memory.to(self.device)
                    adj = tr.adj.to(self.device)
                    te = tr.task_embedding.to(self.device)
                    am = tr.active_mask.to(self.device)
                    sm = tr.src_mask.to(self.device)
                    dm = tr.dst_mask.to(self.device)
                    out = self.policy(
                        mem,
                        adj,
                        te,
                        am,
                        src_mask=sm,
                        dst_mask=dm,
                        forbidden_flat_idx=tr.repeat_penalized_action_idx,
                        forbid_self_loop=self.config.policy_forbid_self_loop,
                        forbid_repeat=self.config.policy_forbid_repeat_action,
                    )
                    n, k = self.policy.num_nodes, self.policy.num_relations
                    term_idx = n * n * k
                    probs = self._mixed_probs(out["all_logits"], tr.mix_alpha)
                    if tr.sampled_from_full:
                        probs, _ = self._terminate_sampling_probs(
                            probs,
                            tr.step_count,
                            term_idx,
                        )
                    lp = action_log_prob(
                        probs,
                        tr.action_flat_idx,
                        n,
                        k,
                        allow_terminate=tr.sampled_from_full,
                        sample_terminate=tr.sampled_from_full,
                    )
                    val = out["value"]
                    ent = interaction_entropy(
                        probs,
                        n,
                        k,
                        allow_terminate=tr.sampled_from_full,
                        sample_terminate=tr.sampled_from_full,
                    )
                    ratio = torch.exp(lp - old_lps[i].to(self.device))
                    a_i = adv[i].to(self.device)
                    clip_r = torch.clamp(
                        ratio,
                        1 - self.config.phase2_clip_eps,
                        1 + self.config.phase2_clip_eps,
                    )
                    pg_loss = pg_loss - torch.min(ratio * a_i, clip_r * a_i)
                    val_loss = val_loss + self.mse(val, rets[i].to(self.device))
                    entropy = entropy + ent
                b = len(mb)
                loss = (
                    pg_loss / b
                    + self.config.phase2_value_coef * val_loss / b
                    - self.config.phase2_entropy_coef * entropy / b
                )
                if not torch.isfinite(loss):
                    continue
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.phase2_max_grad_norm)
                self.optim.step()
                total_pg += (pg_loss / b).item()
                total_val += (val_loss / b).item()
                total_ent += (entropy / b).item()
                n_updates += 1
        if not was_training:
            self.policy.eval()
        n = max(1, n_updates)
        return total_pg / n, total_val / n, total_ent / n
    def save(self, path: str | None = None) -> None:
        path = path or self.config.phase2_save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "rewards": self._episode_rewards}, path)
        print(f"  [Phase2] Saved → {path}")
    @staticmethod
    def load_tasks(path: str, limit: int = 0) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        if p.name == "data2pretrain.jsonl":
            tasks = load_pretrain_tasks(p)
            if limit > 0:
                tasks = tasks[:limit]
            return tasks
        tasks: list[dict] = []
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if limit > 0 and i > limit:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                method = (rec.get("current_method") or rec.get("Conclusion") or "").strip()
                limitation = (rec.get("limitation") or rec.get("Limitations") or "").strip()
                if not method or not limitation or limitation.lower() == "null":
                    continue
                tasks.append({
                    "current_method": method,
                    "limitation": limitation,
                })
        return tasks
