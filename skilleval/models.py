"""Sdílené datové modely a rozhraní mezi moduly (suite, harness, mcp, agent, grading, results).

KONTRAKT: měnit jen zpětně kompatibilně – používají ho všechny části aplikace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------------------------------------------------
# Sada (suite) – výsledek načtení (skilleval.suite.loader)
# ---------------------------------------------------------------------------------------------------------------------

GraderType = Literal["regex", "tool_used", "tool_order", "file_exists", "llm", "baseline"]


class GraderSpec(BaseModel):
    """Jeden grader. `config` = všechny volby typu (pattern, flags, match, target, tool, input_match, min, max, before,
    after, path, exists, criteria, focus, baseline_file) přesně jak jsou v souboru (u llm/baseline z těla → criteria)."""

    model_config = ConfigDict(extra="forbid")
    name: str
    type: GraderType
    weight: float = 1
    arm: Literal["with-only", "both"] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    markdown: str = ""  # graderMarkdown (tělo souboru nebo criteria/pattern z yaml)


class ContextSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scaffold_script: str | None = None  # relativně k case dir
    history_file: str | None = None
    add_dirs: list[str] = Field(default_factory=list)


class FaultSpec(BaseModel):
    """Rozšíření skilleval.yaml – fault injection nad reálným (i mockovaným) serverem."""

    model_config = ConfigDict(extra="forbid")
    tool: str  # název MCP toolu bez prefixu nebo plný mcp__plugin_…
    call: int | None = None  # 1-based pořadí volání toolu (počítají se jen volání odpovídající `when`)
    calls: list[int] | None = None
    every: bool = False
    when: dict[str, Any] | None = None  # podmnožina vstupu (dot-path → hodnota), musí sedět
    error: bool = False
    body: str | None = None  # odpověď místo serveru (náhrady {{input.*}}); None = jen side-effect (`run`)
    before_call: int | None = None  # side-effect před N-tým voláním
    run: list[str] | None = None  # příkaz pro before_call (jen s --hooks), cwd = case dir
    passthrough_after: bool = True


class SimulatedUserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_replies: int = 1
    replies: list[str] = Field(default_factory=list)
    agent: str | None = None  # instrukce pro judge model, jak odpovídat


class ExtensionSpec(BaseModel):
    """<case>/skilleval.yaml – rozšíření, která Claude Code ignoruje."""

    model_config = ConfigDict(extra="forbid")
    mcp_profile: str | None = None
    faults: list[FaultSpec] = Field(default_factory=list)
    setup: list[str] | None = None
    teardown: list[str] | None = None
    simulated_user: SimulatedUserSpec | None = None
    tags_extra: list[str] = Field(default_factory=list)


class CaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    name: str
    dir: Path  # absolutní adresář case
    rel_dir: str  # relativně ke kořeni pluginu ("evals/zapis-bez-dotazu") → JSON `dir`
    source: Literal["prompt", "yaml", "mixed"]
    prompt: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    plugins: list[Path] = Field(default_factory=list)  # absolutní cesty pluginů (z pole `plugins` nebo nejbližší)
    runs: int | None = None
    expected_outcome: str | None = None
    model: str | None = None
    max_turns: int = 10
    timeout_seconds: int = 300
    allowed_tools: list[str] = Field(default_factory=list)
    append_system_prompt: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    context: ContextSpec = Field(default_factory=ContextSpec)
    graders: list[GraderSpec]
    mock_dirs: list[Path] = Field(default_factory=list)  # [<eval>/mocks, <case>/mocks] – existující, pořadí = přednost roste
    extension: ExtensionSpec | None = None

    @property
    def all_tags(self) -> list[str]:
        return list(self.tags) + (list(self.extension.tags_extra) if self.extension else [])


class SkillInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    name: str  # z frontmatteru (nebo název adresáře)
    description: str
    body: str  # SKILL.md bez frontmatteru
    path: Path  # adresář skillu
    plugin: str  # název pluginu

    @property
    def qualified_name(self) -> str:
        return f"{self.plugin}:{self.name}"


class McpServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    plugin: str  # plugin, který server deklaroval (pro pojmenování toolů)
    type: Literal["http", "sse", "stdio"] = "stdio"
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None


class PluginInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    name: str
    root: Path
    skills: list[SkillInfo] = Field(default_factory=list)
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)  # už po expanzi ${VAR} a merge s configem
    eval_dir: Path | None = None


class LoadError(BaseModel):
    path: str
    message: str


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    root: Path  # kořen pluginu (target)
    eval_dir: Path
    plugin: PluginInfo | None  # plugin pod testem (None = nenalezen)
    cases: list[CaseSpec] = Field(default_factory=list)
    load_errors: list[LoadError] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------------------
# Tooly (harness + MCP) – společné rozhraní pro agentní smyčku
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ToolDef:
    name: str  # plný název jako v Claude Code (Read, Skill, mcp__plugin_<plugin>_<server>__<tool>)
    description: str
    input_schema: dict[str, Any]
    kind: Literal["builtin", "mcp"] = "builtin"
    server: str | None = None  # MCP server
    tool: str | None = None  # původní název MCP toolu
    plugin: str | None = None


@dataclass
class ToolOutcome:
    content: str  # text pro model (a do trace tool_result.content)
    is_error: bool = False
    source: str = "builtin"  # builtin | real | mock | agent-mock | replay | fault | denied
    aborted: dict[str, Any] | None = None  # {server, tool, reason} → run končí se skóre 0
    images: list[dict[str, Any]] = field(default_factory=list)  # [{media_type, data(base64)}] – volitelné
    cost_usd: float = 0.0  # náklady (agent mock)


class ToolProvider(Protocol):
    """Implementují BuiltinTools (harness) i McpSession (mcp)."""

    @property
    def tool_defs(self) -> list[ToolDef]: ...

    async def call(self, name: str, args: dict[str, Any]) -> ToolOutcome: ...


@dataclass
class McpCallRecord:
    server: str
    tool: str
    input: dict[str, Any]
    output: str
    is_error: bool
    source: str  # real | mock | agent-mock | replay | fault


# ---------------------------------------------------------------------------------------------------------------------
# Výsledek runu agenta → vstup graderů
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    id: str
    name: str  # plný název (jako v trace)
    input: dict[str, Any]
    output: str | None = None
    is_error: bool = False
    source: str = "builtin"


@dataclass
class RunArtifacts:
    final_message: str  # poslední textová odpověď asistenta
    trace_lines: list[str]  # JSON řádky ve formátu stream-json `claude -p`
    tool_calls: list[ToolCallRecord]
    files_created: list[str]  # relativní cesty k workspace (jen vytvořené během runu)
    workspace: Path
    mock_calls: list[McpCallRecord] = field(default_factory=list)
    turns: int = 0
    error: str | None = None
    aborted: dict[str, Any] | None = None
    api_error: bool = False  # run skončil chybou API modelu → skóre 0 bez hodnocení
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    trace_path: str | None = None


class JudgeVote(BaseModel):
    passed: bool
    explanation: str = ""


class Judge(Protocol):
    """Soudce pro llm/baseline gradery a agent mocky (skilleval.llm)."""

    async def vote(self, system: str, user: str, images: list[dict[str, Any]] | None = None) -> tuple[JudgeVote, float]:
        """Jeden hlas → (verdikt, cena USD)."""
        ...

    async def complete(self, system: str, messages: list[dict[str, str]]) -> tuple[str, float]:
        """Volný text (agent mock, simulovaný uživatel) → (text, cena USD)."""
        ...


@dataclass
class GraderResult:
    name: str
    passed: bool
    weight: float
    explanation: str
    with_only: bool = False
    scored: bool = True
    judge_votes: list[bool] | None = None
    evidence: str | None = None
    judge_cost_usd: float = 0.0

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "passed": self.passed, "weight": self.weight,
                             "explanation": self.explanation, "withOnly": self.with_only, "scored": self.scored}
        if self.judge_votes is not None:
            d["judgeVotes"] = self.judge_votes
        if self.evidence is not None:
            d["evidence"] = self.evidence
        return d


# ---------------------------------------------------------------------------------------------------------------------
# Výsledky (results) – JSON kompatibilní s aggregate-result.json Claude Code (schemaVersion 1)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class RunResult:
    score: float
    passed: bool
    turns: int
    cost_usd: float
    judge_cost_usd: float
    duration_seconds: float
    started_at: str  # ISO 8601 s ms a Z
    error: str | None
    trace_path: str | None
    skipped_paid_graders: bool = False
    aborted: dict[str, Any] | None = None
    graders: list[GraderResult] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)  # mocked:/not granted:/not supported:/fault – jen skilleval
    api_error: bool = False  # run skončil chybou API modelu (skóre 0 bez hodnocení); do JSON se nepíše

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "score": self.score, "passed": self.passed, "turns": self.turns, "costUsd": self.cost_usd,
            "judgeCostUsd": self.judge_cost_usd, "durationSeconds": round(self.duration_seconds),
            "startedAt": self.started_at, "error": self.error, "tracePath": self.trace_path,
            "skippedPaidGraders": self.skipped_paid_graders,
        }
        if self.aborted:
            d["aborted"] = self.aborted
        d["graders"] = [g.to_json() for g in self.graders]
        if self.notices:
            d["notices"] = self.notices
        return d


@dataclass
class CaseResult:
    case: CaseSpec
    runs_per_case: int
    with_runs: list[RunResult] = field(default_factory=list)
    without_runs: list[RunResult] | None = None  # None = single-arm
    single_arm_reason: str | None = None


__all__ = [n for n in dir() if not n.startswith("_")]
