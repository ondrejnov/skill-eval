"""Model agenta: `ChatOpenAI` z `skilleval.llm.make_chat_model` rozšířený o věci OpenRouteru, které ChatOpenAI neumí.

- `reasoning` / `reasoning_details` z odpovědi → `AIMessage.additional_kwargs` (trace `thinking`) a zpět do dalších
  požadavků (`reasoning_details` u assistant zpráv – Anthropic thinking + tool use zachová kontinuitu).
- Prompt caching u `anthropic/*`: breakpoint `cache_control` na poslední zprávě (jako Claude Code – cache roste s
  konverzací; system prompt má vlastní breakpoint už v obsahu SystemMessage). ChatOpenAI u tool zpráv `cache_control`
  z bloků zahazuje, proto se přidává až do hotového payloadu.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

CACHE_CONTROL = {"type": "ephemeral"}


def is_anthropic(slug: str) -> bool:
    return slug.startswith("anthropic/")


def add_cache_breakpoint(message: dict[str, Any]) -> None:
    """Přidá `cache_control` na poslední textový blok zprávy payloadu (string → seznam bloků)."""
    content = message.get("content")
    if isinstance(content, str) and content:
        message["content"] = [{"type": "text", "text": content, "cache_control": CACHE_CONTROL}]
    elif isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                content[i] = {**block, "cache_control": CACHE_CONTROL}
                return


class OpenRouterChatOpenAI(ChatOpenAI):
    """ChatOpenAI + reasoning a prompt caching OpenRouteru. Bez vlastních polí (kvůli záměně třídy instance)."""

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        out = payload.get("messages")
        if not isinstance(out, list):
            return payload
        msgs = self._convert_input(input_).to_messages()
        if len(msgs) == len(out):
            for m, d in zip(msgs, out, strict=True):
                if isinstance(m, AIMessage) and (rd := m.additional_kwargs.get("reasoning_details")):
                    d["reasoning_details"] = rd
        if is_anthropic(self.model_name) and len(out) > 1 and out[-1].get("role") in ("user", "tool"):
            add_cache_breakpoint(out[-1])
        return payload

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump(warnings=False)
        for gen, choice in zip(result.generations, data.get("choices") or [], strict=False):
            msg = (choice or {}).get("message") or {}
            for key in ("reasoning", "reasoning_details"):
                if msg.get(key):
                    gen.message.additional_kwargs[key] = msg[key]
        return result


def upgrade_model(model: Any) -> Any:
    """Instanci přesně `ChatOpenAI` (z make_chat_model) přepne na `OpenRouterChatOpenAI` a dá jí vlastního async HTTP
    klienta. ChatOpenAI jinak sdílí procesně cachovaný httpx klient svázaný s prvním event loopem → v dalším loopu
    (`asyncio.run` víckrát, pytest) „Event loop is closed“. Klienta zavře `aclose_model`. Jiné modely (fake) beze změny."""
    if type(model) is ChatOpenAI:
        model.__class__ = OpenRouterChatOpenAI
        if model.root_async_client is not None:
            model.root_async_client = model.root_async_client.copy(http_client=openai.DefaultAsyncHttpxClient())
            model.async_client = model.root_async_client.chat.completions
    return model


async def aclose_model(model: Any) -> None:
    """Zavře async HTTP klienta modelu vytvořeného přes `upgrade_model`."""
    client = getattr(model, "root_async_client", None) if isinstance(model, OpenRouterChatOpenAI) else None
    if client is not None:
        try:
            await client.close()
        except Exception:  # pragma: no cover - úklid nesmí shodit run
            pass
