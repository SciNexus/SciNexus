from __future__ import annotations
import json
import re
import time
from typing import Any
from openai import OpenAI
from SciNexus.config import ModelConfig
_MD_JSON_FENCE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL | re.IGNORECASE,
)
def extract_json_from_markdown_fence(text: str) -> str | None:
    text = text.strip()
    m = _MD_JSON_FENCE.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    return inner if inner else None
def extract_json_span(text: str) -> str:
    text = text.strip()
    obj_start = text.find("{")
    arr_start = text.find("[")
    if obj_start == -1 and arr_start == -1:
        return text
    if obj_start == -1:
        start, open_ch, close_ch = arr_start, "[", "]"
    elif arr_start == -1:
        start, open_ch, close_ch = obj_start, "{", "}"
    else:
        if arr_start < obj_start:
            start, open_ch, close_ch = arr_start, "[", "]"
        else:
            start, open_ch, close_ch = obj_start, "{", "}"
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]
def extract_json(text: str) -> str:
    text = text.strip()
    fenced = extract_json_from_markdown_fence(text)
    if fenced is not None:
        return extract_json_span(fenced)
    return extract_json_span(text)
class BaseAgent:
    def __init__(self, role: str, config: ModelConfig):
        self.role = role
        self.config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
        max_retries: int = 3,
        json_mode: bool = False,
    ) -> str:
        temp = self.config.llm_temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return self._extract_content(resp)
            except Exception as exc:
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")
    @staticmethod
    def _extract_content(resp: Any) -> str:
        if isinstance(resp, str):
            return resp
        if hasattr(resp, "output_text"):
            return resp.output_text or ""
        if hasattr(resp, "choices") and resp.choices:
            content = resp.choices[0].message.content
            return content if content is not None else ""
        if isinstance(resp, dict):
            choices = resp.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "") or ""
        return str(resp)
    def call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> dict:
        raw = self.call_llm(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        try:
            return json.loads(extract_json(raw))
        except json.JSONDecodeError:
            return {"raw_response": raw}
