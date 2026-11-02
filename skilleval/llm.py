"""OpenRouter modely (ChatOpenAI), výpočet ceny a soudce (Judge) pro llm/baseline gradery a agent mocky."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from skilleval.config import Settings
from skilleval.models import JudgeVote


class ApiError(Exception):
    """Chyba API modelu (HTTP 4xx/5xx po retry) – run končí skóre 0 bez hodnocení."""


def make_chat_model(settings: Settings, slug: str, **overrides: Any):
    """ChatOpenAI nad OpenRouterem. Volby z [models.options] (reasoning_effort, max_tokens, temperature …)."""
    from langchain_openai import ChatOpenAI

    if not settings.api_key:
        raise ApiError("Chybí OPENROUTER_API_KEY (env nebo env_files v skilleval.toml)")
    opts = settings.options_for(slug)
    opts.update(overrides)
    extra_body: dict[str, Any] = {"usage": {"include": True}}
    if effort := opts.pop("reasoning_effort", None):
        extra_body["reasoning"] = {"effort": effort}
    if provider := opts.pop("provider", None):
        extra_body["provider"] = provider
    kwargs: dict[str, Any] = {
        "model": slug,
        "base_url": settings.base_url,
        "api_key": SecretStr(settings.api_key),
        "timeout": opts.pop("timeout", 180),
        "max_retries": opts.pop("max_retries", 3),
        "extra_body": extra_body,
        "default_headers": {"X-Title": "skilleval", "HTTP-Referer": "https://github.com/reactis/skilleval"},
    }
    for k in ("max_tokens", "temperature", "top_p"):
        if k in opts:
            kwargs[k] = opts.pop(k)
    return ChatOpenAI(**kwargs)


def message_cost(settings: Settings, slug: str, msg: BaseMessage) -> float:
    """Cena odpovědi: OpenRouter usage.cost, jinak výpočet z [pricing] (USD / 1M tokenů)."""
    meta = getattr(msg, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(usage, dict) and usage.get("cost") is not None:
        try:
            return float(usage["cost"])
        except (TypeError, ValueError):
            pass
    um = getattr(msg, "usage_metadata", None) or {}
    price = settings.pricing.get(slug)
    if price and um:
        return (um.get("input_tokens", 0) * price.get("input", 0) + um.get("output_tokens", 0) * price.get("output", 0)) / 1e6
    return 0.0


def is_api_error(ex: BaseException) -> bool:
    """HTTP chyba API (openai.APIStatusError/APIConnectionError) – ne chyba našeho kódu."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    return isinstance(ex, (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError, ApiError))


_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b")


def parse_verdict(text: str) -> JudgeVote:
    """Z odpovědi soudce vytáhne JSON {verdict, explanation} nebo aspoň PASS/FAIL (poslední výskyt)."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            verdict = str(data.get("verdict", "")).upper()
            if verdict in ("PASS", "FAIL"):
                return JudgeVote(passed=verdict == "PASS", explanation=str(data.get("explanation", ""))[:2000])
        except (json.JSONDecodeError, AttributeError):
            pass
    found = _VERDICT_RE.findall(text)
    if found:
        return JudgeVote(passed=found[-1] == "PASS", explanation=text.strip()[:2000])
    return JudgeVote(passed=False, explanation="Soudce nevrátil verdikt: " + text.strip()[:500])


class OpenRouterJudge:
    """Implementace protokolu `skilleval.models.Judge`."""

    def __init__(self, settings: Settings, slug: str):
        self.settings = settings
        self.slug = slug
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = make_chat_model(self.settings, self.slug)
        return self._model

    async def vote(self, system: str, user: str, images: list[dict[str, Any]] | None = None) -> tuple[JudgeVote, float]:
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": f"data:{im['media_type']};base64,{im['data']}"}}
                for im in images
            ]
        msg = await self.model.ainvoke([SystemMessage(system), HumanMessage(content=content)])
        text = msg.content if isinstance(msg.content, str) else "".join(
            p.get("text", "") for p in msg.content if isinstance(p, dict))
        return parse_verdict(text), message_cost(self.settings, self.slug, msg)

    async def complete(self, system: str, messages: list[dict[str, str]]) -> tuple[str, float]:
        lc: list[BaseMessage] = [SystemMessage(system)]
        for m in messages:
            lc.append(AIMessage(m["content"]) if m.get("role") == "assistant" else HumanMessage(m["content"]))
        msg = await self.model.ainvoke(lc)
        text = msg.content if isinstance(msg.content, str) else "".join(
            p.get("text", "") for p in msg.content if isinstance(p, dict))
        return text, message_cost(self.settings, self.slug, msg)
