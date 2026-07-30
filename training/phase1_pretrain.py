from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
from SciNexus.config import ModelConfig
from ..graph.evox_policy import EvoXGraphPolicy
from ..hypothesis_text import normalize_hypothesis_text
class TrajectoryDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        config: ModelConfig,
        embed_fn: Callable[[str], np.ndarray],
        embed_cache: dict[str, np.ndarray] | None = None,
    ):
        self.config = config
        self.embed_fn = embed_fn
        self.cache = embed_cache if embed_cache is not None else {}
        self.items: list[dict] = []
        role_to_idx = {r: i for i, r in enumerate(config.agent_roles)}
        type_to_idx = {t: i for i, t in enumerate(config.interaction_types)}
        skip_events = {"reset", "synthesis_checkpoint", "terminal_bonus"}
        for rec in tqdm(records, desc="Embedding trajectories", unit="rec", file=sys.stderr):
            task = rec["task"]
            task_text = (
                f"Current method: {task['current_method']}. "
                f"Limitation: {task['limitation']}."
            )
            task_emb = torch.tensor(self._embed(task_text), dtype=torch.float32)
            steps = []
            for step in rec["trajectory"]:
                event = step.get("event", "step")
                if event in skip_events:
                    continue
                src = step.get("src_role") or step.get("src", "")
                dst = step.get("dst_role") or step.get("dst", "")
                if src == "—" or dst == "—":
                    continue
                if step.get("is_terminate") or event.startswith("terminate"):
                    steps.append({"is_terminate": True})
                    continue
                if src not in role_to_idx or dst not in role_to_idx:
                    continue
                itype = (
                    step.get("interaction_type")
                    or step.get("type")
                    or config.interaction_types[0]
                )
                hyp = normalize_hypothesis_text(
                    step.get("hypothesis") or step.get("hypothesis_snippet", "")
                )
                src_hyp = normalize_hypothesis_text(step.get("src_message", hyp))
                if not hyp or not src_hyp:
                    continue
                steps.append({
                    "src_idx": role_to_idx[src],
                    "dst_idx": role_to_idx[dst],
                    "rel_idx": type_to_idx.get(itype, 0),
                    "is_terminate": False,
                    "src_content": self._embed(src_hyp),
                    "dst_content": self._embed(hyp),
                })
            if steps:
                self.items.append({"task_emb": task_emb, "steps": steps})
    def _embed(self, text: str) -> np.ndarray:
        text = normalize_hypothesis_text(text)
        if not text:
            text = " "
        if text not in self.cache:
            self.cache[text] = self.embed_fn(text)
        return self.cache[text]
    def __len__(self) -> int:
        return len(self.items)
    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]
class Phase1Trainer:
    def __init__(
        self,
        policy: EvoXGraphPolicy,
        config: ModelConfig,
        role_embeddings: torch.Tensor,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.config = config
        self.role_embeddings = role_embeddings.to(device)
        self.device = device
        self.optim = optim.Adam(policy.parameters(), lr=config.phase1_lr)
        self.best_val = float("inf")
        self.patience = 0
    def _flat_target(self, step: dict) -> int:
        n, k = self.policy.num_nodes, self.policy.num_relations
        if step.get("is_terminate"):
            return n * n * k
        idx = step["src_idx"] * n * k + step["dst_idx"] * k + step["rel_idx"]
        return idx
    def _run_trajectory(self, item: dict) -> tuple[torch.Tensor, int]:
        te = item["task_emb"].to(self.device)
        mem = self.policy.init_memory(te, self.role_embeddings, device=self.device)
        adj = torch.zeros(
            self.policy.num_relations,
            self.policy.num_nodes,
            self.policy.num_nodes,
            device=self.device,
        )
        losses = []
        for step in item["steps"]:
            out = self.policy(mem, adj, te)
            target = self._flat_target(step)
            losses.append(F.cross_entropy(out["all_logits"].unsqueeze(0), torch.tensor([target], device=self.device)))
            if step.get("is_terminate"):
                break
            s_ce = torch.tensor(step["src_content"], dtype=torch.float32, device=self.device)
            d_ce = torch.tensor(step["dst_content"], dtype=torch.float32, device=self.device)
            mem = self.policy.update_memory(
                mem,
                step["src_idx"],
                step["dst_idx"],
                step["rel_idx"],
                te,
                s_ce,
                d_ce,
            )
            adj = adj.clone()
            adj[step["rel_idx"], step["dst_idx"], step["src_idx"]] = 1.0
        if not losses:
            return torch.tensor(0.0, device=self.device), 0
        return torch.stack(losses).mean(), len(losses)
    def train(
        self,
        dataset: TrajectoryDataset,
        val_dataset: TrajectoryDataset | None = None,
    ) -> None:
        for epoch in range(1, self.config.phase1_epochs + 1):
            random.shuffle(dataset.items)
            total_loss = 0.0
            n_batches = 0
            batch_loss = torch.tensor(0.0, device=self.device)
            batch_count = 0
            pbar = tqdm(
                dataset.items,
                desc=f"Phase1 train {epoch}/{self.config.phase1_epochs}",
                unit="traj",
                file=sys.stderr,
                dynamic_ncols=True,
            )
            for item in pbar:
                loss, n_steps = self._run_trajectory(item)
                if n_steps == 0:
                    continue
                batch_loss = batch_loss + loss
                batch_count += 1
                if batch_count >= self.config.phase1_batch_size:
                    self.optim.zero_grad()
                    avg_batch = batch_loss / batch_count
                    avg_batch.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optim.step()
                    total_loss += avg_batch.item()
                    n_batches += 1
                    pbar.set_postfix(loss=f"{avg_batch.item():.4f}", refresh=False)
                    batch_loss = torch.tensor(0.0, device=self.device)
                    batch_count = 0
            pbar.close()
            if batch_count > 0:
                self.optim.zero_grad()
                (batch_loss / batch_count).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optim.step()
                total_loss += (batch_loss / batch_count).item()
                n_batches += 1
            avg = total_loss / max(1, n_batches)
            val_avg = self._eval(val_dataset) if val_dataset else None
            msg = f"  [Phase1] epoch {epoch}/{self.config.phase1_epochs}  train_loss={avg:.4f}"
            if val_avg is not None:
                msg += f"  val_loss={val_avg:.4f}"
            print(msg)
            if val_avg is not None:
                if val_avg < self.best_val - self.config.phase1_early_stop_min_delta:
                    self.best_val = val_avg
                    self.patience = 0
                    self.save(self.config.phase1_best_save_path)
                else:
                    self.patience += 1
                    if self.patience >= self.config.phase1_early_stop_patience:
                        print("  [Phase1] Early stopping.")
                        break
        self.save(self.config.phase1_save_path)
    def _eval(self, dataset: TrajectoryDataset) -> float:
        self.policy.eval()
        losses = []
        with torch.no_grad():
            for item in tqdm(
                dataset.items,
                desc="Phase1 val",
                unit="traj",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
            ):
                loss, n = self._run_trajectory(item)
                if n > 0:
                    losses.append(loss.item())
        self.policy.train()
        return sum(losses) / max(1, len(losses))
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict()}, path)
        print(f"  [Phase1] Saved → {path}")
    @staticmethod
    def load_trajectories(path: str) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("trajectory"):
                    records.append(rec)
        return records
