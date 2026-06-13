"""LLM client abstraction.

Supports OpenAI, Anthropic, Ollama (local), NVIDIA NIM (free tier), and a "mock"
provider that produces
a deterministic extractive answer from the retrieved context. The mock provider
exists so the full RAG pipeline runs in CI and on laptops without API keys.
"""

from __future__ import annotations

from typing import Protocol

from echomind_core.config import get_settings
from echomind_core.logging import get_logger

logger = get_logger(__name__)


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 512) -> str: ...


class _MockLLM:
    """Extractive 'answer' built from the most overlapping retrieved sentence."""

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:  # noqa: ARG002
        # the prompt format below puts context after "Context:" and the question after "Question:"
        ctx_marker = "Context:"
        q_marker = "Question:"
        ctx = ""
        question = user
        if ctx_marker in user and q_marker in user:
            after_ctx = user.split(ctx_marker, 1)[1]
            ctx, question_block = after_ctx.split(q_marker, 1)
            question = question_block.strip()

        ctx = ctx.strip()
        if not ctx:
            return "I don't have enough context to answer that."

        q_tokens = {t.lower().strip(".,!?;:") for t in question.split() if len(t) > 3}
        best_score, best_sentence = -1, ""
        for sentence in _split_sentences(ctx):
            s_tokens = {t.lower().strip(".,!?;:") for t in sentence.split() if len(t) > 3}
            score = len(q_tokens & s_tokens)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence or "Based on the available text, no direct answer was found."


class _OpenAILLM:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class _AnthropicLLM:
    def __init__(self, model: str, api_key: str) -> None:
        from anthropic import Anthropic

        self._model = model
        self._client = Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        msg = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        # take the first text block
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""


class _OllamaLLM:
    def __init__(self, model: str, host: str) -> None:
        import httpx

        self._model = model
        self._client = httpx.Client(base_url=host, timeout=60.0)

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        r = self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"num_predict": max_tokens, "temperature": 0.2},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


class _NvidiaLLM:
    """NVIDIA NIM chat completions (OpenAI-compatible, e.g. minimaxai/minimax-m3)."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        import httpx

        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        r = self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 1.00,
                "top_p": 0.95,
                "stream": False,
                "chat_template_kwargs": {"thinking_mode": "disabled"},
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    for ch in text:
        cur.append(ch)
        if ch in ".!?":
            joined = "".join(cur).strip()
            if joined:
                parts.append(joined)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def get_llm() -> LLMClient:
    s = get_settings()
    if s.llm_provider == "openai" and s.openai_api_key:
        logger.info("llm.init", provider="openai", model=s.llm_model)
        return _OpenAILLM(s.llm_model, s.openai_api_key)
    if s.llm_provider == "anthropic" and s.anthropic_api_key:
        logger.info("llm.init", provider="anthropic", model=s.llm_model)
        return _AnthropicLLM(s.llm_model, s.anthropic_api_key)
    if s.llm_provider == "ollama":
        logger.info("llm.init", provider="ollama", model=s.llm_model, host=s.ollama_host)
        return _OllamaLLM(s.llm_model, s.ollama_host)
    if s.llm_provider == "nvidia" and s.nvidia_api_key:
        logger.info("llm.init", provider="nvidia", model=s.llm_model, base_url=s.nvidia_base_url)
        return _NvidiaLLM(s.llm_model, s.nvidia_api_key, s.nvidia_base_url)
    logger.info("llm.init", provider="mock", reason="no api key or provider set")
    return _MockLLM()
