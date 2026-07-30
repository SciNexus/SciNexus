from __future__ import annotations
import asyncio
import re
import httpx
from .prompts import (
    EFFECTIVENESS_PROMPT,
    EFFECTIVENESS_PROMPT_BACKWARD,
    NOVELTY_PROMPT,
    NOVELTY_PROMPT_BACKWARD,
    WIN_RATE_PROMPT,
)
_VOTE_BINARY_RE = re.compile(r"VOTE:\s*([01])\b", re.IGNORECASE)
_VOTE_AB_RE = re.compile(r"VOTE:\s*([AB])\b", re.IGNORECASE)
DEFAULT_JUDGE_MODELS = [
]
JUDGE_SYSTEM = "You are a rigorous scientific reviewer. Follow the user instructions exactly."
def parse_binary_vote(text: str) -> bool:
    matches = _VOTE_BINARY_RE.findall(text or "")
    if not matches:
        return False
    return matches[-1] == "1"
def parse_ab_vote(text: str) -> str | None:
    matches = _VOTE_AB_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].upper()
class LLMJudgeClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        judge_models: list[str] | None = None,
        max_workers: int = 100,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.judge_models = judge_models or list(DEFAULT_JUDGE_MODELS)
        self.sem = asyncio.Semaphore(max_workers)
        self.max_retries = max_retries
    async def _chat(self, model: str, user: str) -> str:
        async with self.sem:
            last_err: Exception | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=180.0) as client:
                        resp = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": JUDGE_SYSTEM},
                                    {"role": "user", "content": user},
                                ],
                                "temperature": 0.0,
                                "max_tokens": 512,
                            },
                        )
                        resp.raise_for_status()
                        return resp.json()["choices"][0]["message"]["content"].strip()
                except Exception as exc:
                    last_err = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(2**attempt)
            return f"[judge_error: {last_err}]"
    async def effectiveness_votes(
        self,
        scientific_context: str,
        generated_hypothesis: str,
    ) -> tuple[int, list[dict]]:
        async def one(model: str, forward: bool) -> dict:
            template = EFFECTIVENESS_PROMPT if forward else EFFECTIVENESS_PROMPT_BACKWARD
            user = template.format(
                scientific_context=scientific_context,
                generated_hypothesis=generated_hypothesis,
            )
            raw = await self._chat(model, user)
            vote = parse_binary_vote(raw)
            return {
                "model": model,
                "forward": forward,
                "vote": int(vote),
                "raw_response": raw,
            }
        details = await asyncio.gather(
            *[one(m, fwd) for m in self.judge_models for fwd in (True, False)]
        )
        votes = sum(d["vote"] for d in details)
        return votes, list(details)
    async def novelty_votes(
        self,
        scientific_context: str,
        generated_hypothesis: str,
    ) -> tuple[int, list[dict]]:
        async def one(model: str, forward: bool) -> dict:
            template = NOVELTY_PROMPT if forward else NOVELTY_PROMPT_BACKWARD
            user = template.format(
                scientific_context=scientific_context,
                generated_hypothesis=generated_hypothesis,
            )
            raw = await self._chat(model, user)
            vote = parse_binary_vote(raw)
            return {
                "model": model,
                "forward": forward,
                "vote": int(vote),
                "raw_response": raw,
            }
        details = await asyncio.gather(
            *[one(m, fwd) for m in self.judge_models for fwd in (True, False)]
        )
        votes = sum(d["vote"] for d in details)
        return votes, list(details)
    async def win_rate_votes(
        self,
        *,
        scientific_context: str,
        hypothesis: str,
        abstract: str,
    ) -> tuple[int, bool, list[dict]]:
        async def one(model: str, hypothesis_first: bool) -> dict:
            if hypothesis_first:
                text_a, text_b = hypothesis, abstract
                hyp_is_a = True
            else:
                text_a, text_b = abstract, hypothesis
                hyp_is_a = False
            user = WIN_RATE_PROMPT.format(
                scientific_context=scientific_context,
                text_A=text_a,
                text_B=text_b,
            )
            raw = await self._chat(model, user)
            choice = parse_ab_vote(raw)
            hyp_wins = (choice == "A" and hyp_is_a) or (choice == "B" and not hyp_is_a)
            return {
                "model": model,
                "hypothesis_first": hypothesis_first,
                "choice": choice,
                "hypothesis_wins": hyp_wins,
                "raw_response": raw,
            }
        details = await asyncio.gather(
            *[one(m, hf) for m in self.judge_models for hf in (True, False)]
        )
        wins = sum(1 for d in details if d["hypothesis_wins"])
        from .metrics import win_is_majority
        return wins, win_is_majority(wins, len(details)), details
