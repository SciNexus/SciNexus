from __future__ import annotations
import math
JUDGE_COUNT = 3
TOTAL_VOTES = JUDGE_COUNT * 2
_PAPER_VOTES = 6
_EFF_GT = 3
_NOV_GT = 4
_WIN_GT = 3
def _passes(votes: int, paper_gt: int, total_votes: int = TOTAL_VOTES) -> bool:
    required = math.ceil((paper_gt + 1) * total_votes / _PAPER_VOTES)
    return votes >= required
def subjective_effectiveness(votes: int, total_votes: int = TOTAL_VOTES) -> float:
    return 100.0 if _passes(votes, _EFF_GT, total_votes) else 0.0
def subjective_novelty(votes: int, total_votes: int = TOTAL_VOTES) -> float:
    return 100.0 if _passes(votes, _NOV_GT, total_votes) else 0.0
def win_is_majority(wins: int, total_votes: int = TOTAL_VOTES) -> bool:
    return _passes(wins, _WIN_GT, total_votes)
def final_effectiveness(obj: float, subj: float) -> float:
    return (obj + subj) / 2.0
def final_novelty(obj: float, subj: float) -> float:
    return (obj + subj) / 2.0
def format_keywords(key_words: list[str] | str | None) -> str:
    if isinstance(key_words, list):
        return ", ".join(str(k).strip() for k in key_words if str(k).strip())
    return str(key_words or "").strip()
