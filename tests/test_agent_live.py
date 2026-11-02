"""Živé testy agenta proti OpenRouteru (opt-in `SKILLEVAL_LIVE=1`, stojí zlomky centu)."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from skilleval.agent.graph import run_agent
from skilleval.config import load_settings
from skilleval.models import ToolDef, ToolOutcome
from skilleval.trace import TraceWriter

pytestmark = pytest.mark.skipif(os.environ.get("SKILLEVAL_LIVE") != "1", reason="live test (SKILLEVAL_LIVE=1)")

LONG_NAME = "mcp__plugin_weather-plugin-with-a-long-name_weather__get_weather_forecast_for_city"


class WeatherProvider:
    def __init__(self, name: str = "get_weather"):
        self.name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def tool_defs(self) -> list[ToolDef]:
        return [ToolDef(name=self.name, description="Get the current weather for a city.", kind="mcp",
                        input_schema={"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Praha'."}},
                                      "required": ["city"]})]

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        self.calls.append(args)
        return ToolOutcome(content=json.dumps({"city": args.get("city"), "temperature_c": 21, "sky": "clear"}),
                           source="real")


def _trace(slug: str) -> TraceWriter:
    return TraceWriter(cwd="/tmp", model=slug, tools=[], mcp_servers=[], skills=[], plugins=[])


async def test_live_weather_tool_call():
    settings = load_settings()
    slug = settings.resolve_model("haiku")
    prov = WeatherProvider()
    trace = _trace(slug)
    out = await run_agent(settings=settings, model_slug=slug, system_prompt="You are a helpful assistant. Use tools.",
                          prompt="Jaké je teď počasí v Praze? Použij tool.", providers=[prov], trace=trace,
                          max_turns=5, timeout_seconds=120)
    print(out, *trace.lines, sep="\n")
    assert out.error is None, out.error
    assert prov.calls and "prah" in str(prov.calls[0].get("city", "")).lower()
    assert [c.name for c in out.tool_calls] == ["get_weather"]
    assert "21" in out.final_message
    assert 0 < out.cost_usd < 0.05
    rows = [json.loads(line) for line in trace.lines]
    assert rows[-1]["type"] == "result" and rows[-1]["total_cost_usd"] == pytest.approx(out.cost_usd)
    assert any(r["type"] == "assistant" and r["message"]["content"][0]["type"] == "tool_use" for r in rows)


async def test_live_prompt_cache_reasoning_and_alias():
    """Dlouhý system prompt (nad minimum cache Haiku) → tah 2 čte cache; reasoning_details projdou zpět; alias názvu."""
    settings = load_settings()
    slug = settings.resolve_model("haiku")
    settings.model_options[slug] = {"reasoning_effort": "high"}
    system = "You are a weather assistant. Always use the tool before answering.\n" + "\n".join(
        f"Rule {i}: city code C{i:04d} maps to region R{i % 17} and must never be confused with C{i + 1:04d}."
        for i in range(500))
    prov = WeatherProvider(LONG_NAME)
    trace = _trace(slug)
    out = await run_agent(settings=settings, model_slug=slug, system_prompt=system,
                          prompt="What's the weather in Brno? Think about it, then call the tool.", providers=[prov],
                          trace=trace, max_turns=5, timeout_seconds=180)
    print(out, *trace.lines, sep="\n")
    assert out.error is None, out.error
    assert [c.name for c in out.tool_calls] == [LONG_NAME]
    usages = [json.loads(line)["message"]["usage"] for line in trace.lines if json.loads(line)["type"] == "assistant"]
    assert usages[0].get("cache_creation_input_tokens", 0) > 0 or usages[0].get("cache_read_input_tokens", 0) > 0
    assert usages[-1].get("cache_read_input_tokens", 0) > 0


class ManyToolsProvider:
    """40 MCP-like toolů (webstore styl); orders_get vrací data, ostatní obecnou odpověď."""

    PREFIX = "mcp__plugin_webstore-core_webstore__"
    TOOLS = ["core_whoami", "core_search", "orders_get", "orders_list", "orders_update", "orders_set_status",
             "orders_copy", "orders_delete", "orders_items_state", "orders_customer_history", "invoices_get",
             "invoices_list", "invoices_create", "invoices_update", "invoices_delete", "invoices_settings_get",
             "deliverylists_get", "deliverylists_list", "deliverylists_create_from_order", "deliverylists_delete",
             "store_list", "store_summary", "store_receive", "store_reserve_item", "store_free_item", "store_pieces",
             "store_product_availability", "store_min_stock", "shipping_track", "shipping_branches",
             "shipping_add_packages", "shipping_waiting", "analytics_orders_kpi", "analytics_sales_kpi",
             "analytics_receivables", "analytics_store_overview", "customers_get", "customers_list",
             "localization_list", "analytics_documents_consistency_check_for_all_documents"]

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def tool_defs(self) -> list[ToolDef]:
        return [ToolDef(name=self.PREFIX + t, kind="mcp", server="webstore", tool=t, plugin="webstore-core",
                        description=f"{t.replace('_', ' ').capitalize()} – nástroj administrace Webstore.",
                        input_schema={"type": "object", "properties": {
                            "id": {"type": "integer", "description": "ID záznamu, např. 71781."}}})
                for t in self.TOOLS]

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        self.calls.append((name, args))
        if name.endswith("__orders_get"):
            return ToolOutcome(content=json.dumps({"id": args.get("id"), "status": {"id": 2, "label": "Vyřizuje se"}},
                                                  ensure_ascii=False), source="real")
        return ToolOutcome(content="{}", source="real")


async def test_live_deferred_tools_toolsearch():
    settings = load_settings()
    slug = settings.resolve_model("haiku")
    prov = ManyToolsProvider()
    trace = _trace(slug)
    out = await run_agent(settings=settings, model_slug=slug,
                          system_prompt="You are an assistant for the Webstore e-shop administration.",
                          prompt="V jakém stavu je objednávka 71781?", providers=[prov], trace=trace,
                          max_turns=6, timeout_seconds=180, defer_tools=True)
    print(out, *trace.lines, sep="\n")
    assert out.error is None, out.error
    names = [c.name for c in out.tool_calls]
    assert names[0] == "ToolSearch"
    assert ManyToolsProvider.PREFIX + "orders_get" in names
    assert "vyřizuje" in out.final_message.lower()
    refs = [json.loads(line)["message"]["content"][0]["content"] for line in trace.lines
            if json.loads(line)["type"] == "user" and isinstance(json.loads(line)["message"]["content"], list)]
    assert isinstance(refs[0], list) and refs[0][0]["type"] == "tool_reference"
