"""Fault injection (rozšíření `skilleval.yaml`) – platí v každém režimu, před mockem i serverem.

Každý fault má vlastní počítadlo volání; počítají se jen volání jeho toolu, jejichž vstup odpovídá `when`.
- odpověď (`body`): na `call: N`, `calls: [..]`, `every: true`; bez selektoru = 1. volání (případně `before_call`);
  `passthrough_after: false` = po prvním odpálení odpovídá fault i na všechna další volání,
- side-effect (`before_call: N` + `run`): příkaz se spustí před N-tým voláním (jen s `--hooks`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from skilleval.mcp.expect import subset_match
from skilleval.models import FaultSpec


@dataclass
class FaultHit:
    index: int  # pořadí faultu v skilleval.yaml (0-based)
    fault: FaultSpec
    kind: Literal["response", "run"]
    call: int  # pořadí volání pro tento fault (1-based)


def fault_tool_matches(fault: FaultSpec, *, tool: str, full_name: str) -> bool:
    """Tool faultu bez prefixu (`orders_update`) nebo plný název (`mcp__plugin_…__orders_update`)."""
    return fault.tool in (tool, full_name)


class FaultEngine:
    def __init__(self, faults: list[FaultSpec] | None):
        self.faults = list(faults or [])
        self.counters = [0] * len(self.faults)
        self.fired = [False] * len(self.faults)

    def __bool__(self) -> bool:
        return bool(self.faults)

    @staticmethod
    def _selected(f: FaultSpec, n: int) -> bool:
        if f.every:
            return True
        if f.calls:
            return n in f.calls
        if f.call is not None:
            return n == f.call
        return n == (f.before_call or 1)

    def on_call(self, *, tool: str, full_name: str, input: dict[str, Any]) -> tuple[list[FaultHit], FaultHit | None]:
        """Zaregistruje volání → (side-effecty ke spuštění před voláním, odpověď faultu nebo None)."""
        runs: list[FaultHit] = []
        response: FaultHit | None = None
        for i, f in enumerate(self.faults):
            if not fault_tool_matches(f, tool=tool, full_name=full_name) or not subset_match(f.when, input):
                continue
            self.counters[i] += 1
            n = self.counters[i]
            if f.before_call is not None and f.before_call == n and f.run:
                runs.append(FaultHit(i, f, "run", n))
            if f.body is None or response is not None:
                continue
            if self._selected(f, n) or (self.fired[i] and not f.passthrough_after):
                self.fired[i] = True
                response = FaultHit(i, f, "response", n)
        return runs, response
