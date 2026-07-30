from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from .config import ModelConfig
from .hypothesis_text import normalize_hypothesis_text
@dataclass(frozen=True)
class CompResult:
    score_a: float
    score_b: float
    margin: float
class ComparatorLike(Protocol):
    def compare_scores(
        self,
        task: dict,
        hyp_a: str,
        hyp_b: str,
        use_cache: bool = True,
    ) -> tuple[float, float, float]: ...
    def compare_margin(
        self,
        task: dict,
        hyp_a: str,
        hyp_b: str,
        use_cache: bool = True,
    ) -> float: ...
def comp_pair(
    comparator: ComparatorLike | Callable[..., Any],
    task: dict,
    hyp_a: str,
    hyp_b: str,
) -> CompResult:
    if not hyp_a or not hyp_b:
        return CompResult(0.0, 0.0, 0.0)
    if hasattr(comparator, "compare_scores"):
        score_a, score_b, margin = comparator.compare_scores(task, hyp_a, hyp_b)
        return CompResult(float(score_a), float(score_b), float(margin))
    if hasattr(comparator, "compare_margin"):
        margin = float(comparator.compare_margin(task, hyp_a, hyp_b))
        return CompResult(0.0, 0.0, margin)
    result = float(comparator(task, hyp_a, hyp_b))
    return CompResult(0.0, 0.0, result)
def comp_margin(
    comparator: ComparatorLike | Callable[..., Any],
    task: dict,
    hyp_a: str,
    hyp_b: str,
) -> float:
    return comp_pair(comparator, task, hyp_a, hyp_b).margin
def step_reward(
    comparator: ComparatorLike | Callable[..., Any],
    task: dict,
    new_hyp: str,
    src_hyp: str,
    best_hyp: str,
    config: ModelConfig,
    step_t: int = 1,
) -> tuple[float, CompResult, CompResult]:
    local = comp_pair(comparator, task, new_hyp, src_hyp)
    global_comp = comp_pair(comparator, task, new_hyp, best_hyp)
    c_global = max(0.0, global_comp.margin)
    if not config.use_step_reward:
        return 0.0, local, global_comp
    margin = (
        config.lambda_local * local.margin
        + config.lambda_global * c_global
        - config.lambda_step * float(step_t)
    )
    tau_s = max(float(config.step_tau), 1e-6)
    reward = config.reward_step_scale * math.tanh(margin / tau_s)
    return reward, local, global_comp
def terminal_reward(
    comparator: ComparatorLike | Callable[..., Any],
    task: dict,
    final_hyp: str,
    best_hyp: str,
    seed_hyp: str,
    config: ModelConfig,
) -> tuple[float, CompResult, CompResult]:
    if not final_hyp or not seed_hyp or not config.use_terminal_reward:
        zero = CompResult(0.0, 0.0, 0.0)
        return 0.0, zero, zero
    episode_best = best_hyp or final_hyp
    comp_base = comp_pair(comparator, task, final_hyp, seed_hyp)
    if normalize_hypothesis_text(final_hyp) == normalize_hypothesis_text(episode_best):
        comp_best = comp_base
    else:
        comp_best = comp_pair(comparator, task, final_hyp, episode_best)
    alpha = config.terminal_alpha
    margin = alpha * comp_base.margin + (1.0 - alpha) * comp_best.margin
    tau = max(float(config.terminal_tau), 1e-6)
    reward = config.reward_terminal_scale * math.tanh(margin / tau)
    return reward, comp_base, comp_best
