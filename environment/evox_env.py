from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional
import numpy as np
import torch
from SciNexus.config import ModelConfig
from ..agents import create_default_agents
from ..graph.graph_state import GraphState
from ..graph.evox_policy import EvoXGraphPolicy
from ..pretrain_data.task_utils import method_to_h0
from ..reward import CompResult, step_reward, terminal_reward
@dataclass
class StepResult:
    node_features: torch.Tensor
    adj: torch.Tensor
    task_embedding: torch.Tensor
    active_mask: torch.Tensor
    memory: Optional[torch.Tensor]
    reward: float
    done: bool
    term_prob: float
    info: dict
def _comp_info(prefix: str, comp: CompResult) -> dict[str, float]:
    return {
        f"{prefix}_score_a": comp.score_a,
        f"{prefix}_score_b": comp.score_b,
        f"{prefix}_margin": comp.margin,
    }
class EvoXEnv:
    def __init__(
        self,
        config: ModelConfig,
        embed_fn: Callable[[str], np.ndarray],
        comparator_fn: Optional[Any] = None,
        policy: Optional[EvoXGraphPolicy] = None,
    ):
        self.config = config
        self.embed_fn = embed_fn
        self.comparator = comparator_fn
        self.policy = policy
        self.agents = create_default_agents(config)
        self.graph = GraphState(
            agent_roles=config.agent_roles,
            interaction_types=config.interaction_types,
            text_embed_dim=config.text_embed_dim,
            role_embed_dim=config.role_embed_dim,
        )
        self._task: dict = {}
        self._task_emb = np.zeros(config.text_embed_dim)
        self._current_best_hyp = ""
        self._last_hypothesis = ""
        self._seed_hyp = ""
        self._recent_quality_scores: list[int] = []
        self._episode_log: list[dict] = []
        self._policy_memory: Optional[torch.Tensor] = None
        self._role_emb_matrix: Optional[torch.Tensor] = None
    def _policy_device(self) -> torch.device:
        if self.policy is None:
            return torch.device("cpu")
        return next(self.policy.parameters()).device
    def _build_role_embeddings(self) -> torch.Tensor:
        rows = [
            self.graph.get_role_embedding(i).numpy()
            for i in range(self.graph.num_nodes)
        ]
        return torch.tensor(np.stack(rows), dtype=torch.float32)
    @staticmethod
    def _extract_initial_hypothesis(task: dict) -> str:
        method = task.get("current_method")
        if method is not None and str(method).strip():
            return method_to_h0(str(method))
        raise ValueError(
            "task must include 'current_method' to derive seed baseline h_0."
        )
    def reset(self, task: dict) -> StepResult:
        self._task = task
        self._recent_quality_scores = []
        self._episode_log = []
        for agent in self.agents.values():
            agent.reset()
        self.graph.reset()
        task_text = (
            f"Current method: {task['current_method']}. "
            f"Limitation: {task['limitation']}."
        )
        self._task_emb = self.embed_fn(task_text)
        self._role_emb_matrix = self._build_role_embeddings()
        if self.policy is not None:
            te = torch.tensor(self._task_emb, dtype=torch.float32, device=self._policy_device())
            re = self._role_emb_matrix.to(self._policy_device())
            self._policy_memory = self.policy.init_memory(te, re, device=self._policy_device())
        seed_hyp = self._extract_initial_hypothesis(task)
        self._seed_hyp = seed_hyp
        self._current_best_hyp = seed_hyp
        self._last_hypothesis = seed_hyp
        self._log("reset", "—", "—", "—", seed_hyp, 0.0)
        return self._obs(reward=0.0, done=False, term_prob=0.0, info={"event": "reset"})
    def step(self, action: dict, term_prob: float = 0.0) -> StepResult:
        if action["is_terminate"]:
            final_hyp = self._last_hypothesis or self._seed_hyp
            best = self._current_best_hyp or self.graph.get_best_hypothesis()
            step_count = self.graph.current_step
            term_r, comp_base, comp_best = (
                terminal_reward(
                    self.comparator,
                    self._task,
                    final_hyp,
                    best,
                    self._seed_hyp,
                    self.config,
                )
                if self.comparator
                else (0.0, CompResult(0.0, 0.0, 0.0), CompResult(0.0, 0.0, 0.0))
            )
            self._log(
                "terminate_action", "—", "—", "terminate", final_hyp, term_r,
                comp_base=comp_base, comp_global=comp_best,
            )
            info = {
                "event": "terminate_action",
                "final_hypothesis": final_hyp,
                "best_hypothesis": best,
                "step_reward": 0.0,
                "terminal_reward": term_r,
                "step_count": step_count,
                **_comp_info("terminal_base", comp_base),
                **_comp_info("terminal_best", comp_best),
            }
            return self._obs(reward=term_r, done=True, term_prob=term_prob, info=info)
        src_role = self.config.agent_roles[action["src"]]
        dst_role = self.config.agent_roles[action["dst"]]
        itype = self.config.interaction_types[action["type"]]
        src_agent = self.agents[src_role]
        dst_agent = self.agents[dst_role]
        src_hyp = self._resolve_src_hypothesis(src_role, src_agent)
        new_hyp = dst_agent.respond_to_interaction(
            task=self._task,
            src_role=src_role,
            src_message=src_hyp,
            interaction_type=itype,
        )
        new_emb = self.embed_fn(new_hyp)
        src_emb = self.embed_fn(src_hyp)
        self.graph.add_edge(
            src_role=src_role,
            dst_role=dst_role,
            interaction_type=itype,
            src_message=src_hyp,
            dst_response=new_hyp,
            dst_hypothesis_emb=new_emb,
        )
        if self.policy is not None and self._policy_memory is not None:
            te = torch.tensor(self._task_emb, dtype=torch.float32, device=self._policy_device())
            s_ce = torch.tensor(src_emb, dtype=torch.float32, device=self._policy_device())
            d_ce = torch.tensor(new_emb, dtype=torch.float32, device=self._policy_device())
            si = self.config.agent_roles.index(src_role)
            di = self.config.agent_roles.index(dst_role)
            ri = self.config.interaction_types.index(itype)
            with torch.no_grad():
                self._policy_memory = self.policy.update_memory(
                    self._policy_memory, si, di, ri, te, s_ce, d_ce
                ).detach()
        if self.comparator:
            r, local_comp, global_comp = step_reward(
                self.comparator,
                self._task,
                new_hyp,
                src_hyp,
                self._current_best_hyp,
                self.config,
                step_t=self.graph.current_step,
            )
            self._recent_quality_scores.append(1 if global_comp.margin > 0 else 0)
            if global_comp.margin > 0:
                self._current_best_hyp = new_hyp
        else:
            r = 0.0
            local_comp = CompResult(0.0, 0.0, 0.0)
            global_comp = CompResult(0.0, 0.0, 0.0)
            self._recent_quality_scores.append(0)
        self._last_hypothesis = new_hyp
        done = False
        event = "step"
        if self.graph.current_step >= self.config.max_steps:
            done, event = True, "max_steps"
        step_info = {
            "event": event,
            "step_reward": r,
            **_comp_info("local", local_comp),
            **_comp_info("global", global_comp),
        }
        if done:
            final_hyp = new_hyp
            best = self._current_best_hyp or self.graph.get_best_hypothesis()
            step_count = self.graph.current_step
            term_r, comp_base, comp_best = (
                terminal_reward(
                    self.comparator,
                    self._task,
                    final_hyp,
                    best,
                    self._seed_hyp,
                    self.config,
                )
                if self.comparator
                else (0.0, CompResult(0.0, 0.0, 0.0), CompResult(0.0, 0.0, 0.0))
            )
            total = r + term_r
            self._log(
                event, src_role, dst_role, itype, new_hyp, total,
                comp_local=local_comp, comp_global=global_comp, src_message=src_hyp,
            )
            self._log(
                "terminal_bonus", "—", "—", "terminal", final_hyp, term_r,
                comp_base=comp_base, comp_best=comp_best,
            )
            info = {
                **step_info,
                "final_hypothesis": final_hyp,
                "best_hypothesis": best,
                "terminal_reward": term_r,
                "step_count": step_count,
                **_comp_info("terminal_base", comp_base),
                **_comp_info("terminal_best", comp_best),
            }
            return self._obs(reward=total, done=True, term_prob=term_prob, info=info)
        self._log(
            event, src_role, dst_role, itype, new_hyp, r,
            comp_local=local_comp, comp_global=global_comp, src_message=src_hyp,
        )
        return self._obs(
            reward=r,
            done=False,
            term_prob=term_prob,
            info={**step_info, "terminal_reward": 0.0},
        )
    def _resolve_src_hypothesis(self, src_role: str, src_agent) -> str:
        if src_agent.current_hypothesis:
            return src_agent.current_hypothesis
        src_node = self.graph.nodes[self.graph.role_to_idx[src_role]]
        if src_node.current_hypothesis:
            src_agent.current_hypothesis = src_node.current_hypothesis
            return src_node.current_hypothesis
        hyp = src_agent.generate_initial_hypothesis(self._task)
        hyp_emb = self.embed_fn(hyp)
        self.graph.activate_node(src_role, hyp, hyp_emb)
        return hyp
    def _obs(
        self,
        reward: float,
        done: bool,
        term_prob: float,
        info: dict,
    ) -> StepResult:
        mem = (
            self._policy_memory.detach().clone()
            if self._policy_memory is not None
            else None
        )
        return StepResult(
            node_features=self.graph.get_node_features(),
            adj=self.graph.adj.clone(),
            task_embedding=torch.tensor(self._task_emb, dtype=torch.float32),
            active_mask=self.graph.get_active_mask(),
            memory=mem,
            reward=reward,
            done=done,
            term_prob=term_prob,
            info=info,
        )
    def get_action_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        active = self.graph.get_active_mask()
        all_ok = torch.ones(len(self.config.agent_roles), dtype=torch.bool)
        if not active.any():
            return all_ok, all_ok
        return active, all_ok
    def _log(
        self,
        event: str,
        src: str,
        dst: str,
        itype: str,
        hypothesis: str,
        reward: float,
        comp_local: CompResult | None = None,
        comp_global: CompResult | None = None,
        comp_base: CompResult | None = None,
        comp_best: CompResult | None = None,
        src_message: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "step": self.graph.current_step,
            "event": event,
            "src": src,
            "dst": dst,
            "type": itype,
            "hypothesis": hypothesis,
            "hypothesis_snippet": hypothesis[:200],
            "src_message": src_message or "",
            "reward": reward,
        }
        if comp_local is not None:
            entry.update(_comp_info("local", comp_local))
        if comp_global is not None:
            entry.update(_comp_info("global", comp_global))
        if comp_base is not None:
            entry.update(_comp_info("terminal_base", comp_base))
        if comp_best is not None:
            entry.update(_comp_info("terminal_best", comp_best))
        self._episode_log.append(entry)
    @property
    def episode_log(self) -> list[dict]:
        return self._episode_log
