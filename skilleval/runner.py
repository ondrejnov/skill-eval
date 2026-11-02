"""Orchestrace běhu sady: cases × arms × runs, souběh, limity, výsledky a exit kódy."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TextIO

from skilleval import __version__
from skilleval.agent.deferred import deferred_tool_names
from skilleval.agent.graph import run_agent
from skilleval.agent.history import load_history
from skilleval.agent.simulated_user import make_ask_user_hook
from skilleval.config import ConfigError, Settings, deep_merge, load_settings
from skilleval.grading import excluded_graders, grade_run
from skilleval.harness import sandbox
from skilleval.harness.prompt import build_system_prompt
from skilleval.harness.tools_builtin import BuiltinTools
from skilleval.llm import OpenRouterJudge
from skilleval.mcp.mocks import write_adopt
from skilleval.mcp.session import McpSession
from skilleval.models import (
    CaseResult,
    CaseSpec,
    Judge,
    McpServerConfig,
    PluginInfo,
    RunArtifacts,
    RunResult,
    Suite,
)
from skilleval.plugin import find_plugin_root, load_plugin
from skilleval.results import (
    build_aggregate,
    case_passed,
    default_results_dir,
    progress_line,
    render_report,
    summary_table,
    write_results,
)
from skilleval.suite.loader import load_suite
from skilleval.trace import TraceWriter

# Tooly, které Claude Code nabízí i bez `allowed_tools` (dle trace `claude plugin eval`: Glob, Grep, Read, Skill …)
DEFAULT_TOOLS = ["Skill", "Read", "Glob", "Grep"]
# Read-only tooly, které case smí vyžádat přes allowed_tools bez --allow-tools
READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "NotebookRead", "Skill", "AskUserQuestion", "Agent", "TodoWrite",
                   "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop"}


@dataclass
class RunOptions:
    target: str | None = None
    case_globs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    eval_dir: str | None = None
    extra_plugins: list[str] = field(default_factory=list)
    runs: int | None = None
    concurrency: int = 1
    model: str | None = None
    judge_model: str | None = None
    ablation: str | None = None  # None = per case, "none" | "with-without"
    threshold: float = 1.0
    max_cost_usd: float | None = None
    allow_tools: list[str] = field(default_factory=list)
    strict_grants: bool = False
    mocks: str = "record"
    allow_real_servers: bool = False
    scaffold: bool = False
    hooks: bool = False
    mcp_config: str | None = None
    mcp_profile: str | None = None
    config: str | None = None
    env_files: list[str] = field(default_factory=list)
    json_out: str | None = None  # "-" = stdout
    output_dir: str | None = None
    keep_temp: bool = False
    verbose: bool = False
    no_report: bool = False
    color: bool | None = None
    defer_tools: str | None = None  # auto | on | off (přebíjí [limits] defer_tools)


class AbortSuite(Exception):
    """Ukončení celé sady (auth chyba OpenRouteru hned na začátku)."""


def _now_iso() -> str:
    d = datetime.now(UTC)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def load_mcp_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("mcpServers", data)


class Runner:
    def __init__(self, opts: RunOptions, *, out: TextIO | None = None, err: TextIO | None = None,
                 model_factory: Callable[[CaseSpec, str, int], Any] | None = None, judge: Judge | None = None):
        self.opts = opts
        self.model_factory = model_factory  # testy: fake model (case, arm, run)
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.quiet = opts.json_out is not None
        self.color = opts.color if opts.color is not None else self.err.isatty()
        self.settings: Settings | None = None
        self.suite: Suite | None = None
        self.results: dict[str, CaseResult] = {}
        self.spent = 0.0
        self.partial_reason: str | None = None
        self.api_errors = 0
        self.runs_started = 0
        self.runs_ok = 0
        self.results_dir: Path | None = None
        self.started = datetime.now(UTC)
        self.started_iso = _now_iso()
        self.t0 = time.monotonic()
        self._plugins: dict[tuple[str, str | None], PluginInfo] = {}
        self._judge: Judge | None = judge
        self.exit_code = 0

    # --- výstup -------------------------------------------------------------------------------------------------------

    def log(self, msg: str) -> None:
        if not self.quiet or self.opts.verbose:
            print(msg, file=self.err, flush=True)

    def warn(self, msg: str) -> None:
        print(msg, file=self.err, flush=True)

    # --- příprava -----------------------------------------------------------------------------------------------------

    def prepare(self) -> Suite:
        o = self.opts
        target_path = Path(o.target).resolve() if o.target else Path.cwd()
        root = find_plugin_root(target_path if target_path.is_dir() else target_path.parent)
        self.settings = s = load_settings(o.config, o.env_files, plugin_root=root)
        if o.defer_tools:
            s.defer_tools = o.defer_tools
        if o.mcp_config:
            s.mcp_servers = deep_merge(s.mcp_servers, load_mcp_config(o.mcp_config))
        warnings: list[str] = []
        self.suite = load_suite(o.target, s, eval_dir=o.eval_dir, case_globs=o.case_globs or None,
                                tags=o.tags or None, profile=o.mcp_profile, resolve_mcp=False, warnings=warnings)
        for w in warnings:
            self.warn(f"Warning: {w}")
        return self.suite

    def plugin(self, path: Path, profile: str | None) -> PluginInfo:
        key = (str(path), profile)
        if key not in self._plugins:
            assert self.settings is not None
            self._plugins[key] = load_plugin(path, self.settings, profile=profile, eval_dir=self.opts.eval_dir)
        return self._plugins[key]

    def case_plugins(self, case: CaseSpec) -> list[PluginInfo]:
        profile = (case.extension.mcp_profile if case.extension and case.extension.mcp_profile
                   else self.opts.mcp_profile)
        paths = list(case.plugins) + [Path(p).resolve() for p in self.opts.extra_plugins]
        out: list[PluginInfo] = []
        seen: set[Path] = set()
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            out.append(self.plugin(p, profile))
        return out

    @property
    def judge(self) -> Judge:
        if self._judge is None:
            assert self.settings is not None
            self._judge = OpenRouterJudge(self.settings, self.settings.resolve_model(self.opts.judge_model, judge=True))
        return self._judge

    def arms_for(self, case: CaseSpec, plugins: list[PluginInfo]) -> tuple[list[str], str | None]:
        abl = self.opts.ablation
        if abl == "none":
            return ["with"], None
        if not plugins:
            return ["with"], "no plugin resolved"
        if abl is None and case.context.history_file:
            return ["with"], "history_file (single-arm, no Δ)"
        return ["with", "without"], None

    def agent_model(self, case: CaseSpec) -> str:
        assert self.settings is not None
        return self.settings.resolve_model(self.opts.model or case.model)

    # --- jeden run ----------------------------------------------------------------------------------------------------

    def _tools_for(self, case: CaseSpec, with_plugin: bool, has_skills: bool) -> tuple[list[str], list[str]]:
        """→ (požadované builtin tooly, notices). MCP granty řeší McpSession."""
        tools: list[str] = [t for t in DEFAULT_TOOLS if t != "Skill" or (with_plugin and has_skills)]
        notices: list[str] = []
        for t in case.allowed_tools:
            base = t.split("(", 1)[0]
            if base.startswith("mcp__"):
                continue
            if base in READ_ONLY_TOOLS or any(base == a.split("(", 1)[0] for a in self.opts.allow_tools):
                if t not in tools:
                    tools.append(t)
            else:
                notices.append(f"not granted: {t}")
        for t in self.opts.allow_tools:
            if not t.startswith("mcp__") and t not in tools:
                tools.append(t)
        return tools, notices

    async def execute_run(self, case: CaseSpec, arm: str, idx: int, total: int, plugins: list[PluginInfo],
                          excluded: set[str], out_dir: Path) -> RunResult:
        assert self.settings is not None and self.suite is not None
        s, o = self.settings, self.opts
        with_plugin = arm == "with"
        started_iso = _now_iso()
        t0 = time.monotonic()
        slug = self.agent_model(case)
        dirs = sandbox.make_run_dirs(o.keep_temp)
        notices: list[str] = []
        ext = case.extension
        run_env = {**sandbox.allowlisted_env(), **case.env, "HOME": str(dirs.home)}
        hook_env = {**s.env, "SKILLEVAL_RUN_DIR": str(dirs.root), "SKILLEVAL_CASE": case.name,
                    "SKILLEVAL_ARM": arm, "SKILLEVAL_RUN": str(idx), **case.env}

        def fail(error: str, *, api: bool = False) -> RunResult:
            return RunResult(score=0.0, passed=False, turns=0, cost_usd=0.0, judge_cost_usd=0.0,
                             duration_seconds=time.monotonic() - t0, started_at=started_iso, error=error,
                             trace_path=None, notices=notices, api_error=api)

        try:
            if ext and ext.setup:
                if not o.hooks:
                    notices.append("setup skipped: --hooks off")
                elif err := await sandbox.run_hook(ext.setup, case=case, env=hook_env):
                    return fail(f"setup failed: {err}")
            if case.context.scaffold_script:
                if o.scaffold:
                    if err := await sandbox.run_scaffold(case, dirs):
                        return fail(err)
                else:
                    notices.append("scaffold skipped: --scaffold off")

            skills = [sk for p in plugins for sk in p.skills] if with_plugin else []
            servers: list[McpServerConfig] = []
            if with_plugin:
                seen: dict[str, McpServerConfig] = {}
                for p in plugins:
                    for name, cfg in p.mcp_servers.items():
                        if name in seen:
                            if seen[name].model_dump() != cfg.model_dump():
                                notices.append(f"server {name}: duplicate in {p.name}, using {seen[name].plugin}")
                            continue
                        seen[name] = cfg
                servers = list(seen.values())

            tools, tnotices = self._tools_for(case, with_plugin, bool(skills))
            notices += tnotices
            readable = [p.root for p in plugins] + [case.dir / d for d in case.context.add_dirs]
            forbidden = [self.suite.eval_dir] + [p.eval_dir for p in plugins if p.eval_dir]
            sim = make_ask_user_hook(ext.simulated_user, self.judge) if ext and ext.simulated_user else None
            builtin = BuiltinTools(workspace=dirs.workspace, readable=readable if with_plugin else
                                   [case.dir / d for d in case.context.add_dirs], forbidden=forbidden,
                                   skills=skills, tools=tools, max_result_chars=s.max_tool_result_chars, ask_user=sim)
            notices += [f"not supported: {t}" for t in builtin.unsupported]
            builtin.snapshot()

            trace_holder: dict[str, TraceWriter] = {}

            def on_event(kind: str, data: dict[str, Any]) -> None:
                if tw := trace_holder.get("t"):
                    tw.system(kind, **data)

            rec_dir = (out_dir / "mock-recordings") if o.mocks == "record" else None
            default_plugin = plugins[0].name if plugins else None
            async with McpSession(servers if with_plugin else [], case=case,
                                  mode="off" if o.mocks == "off" else "record",
                                  allow_real=o.allow_real_servers, allow_tools=o.allow_tools,
                                  strict_grants=o.strict_grants, judge=self.judge, settings=s, hooks=o.hooks,
                                  recordings_dir=rec_dir, run_env=run_env, on_event=on_event,
                                  default_plugin=default_plugin if with_plugin else None) as mcp:
                notices += mcp.notices
                tool_defs = builtin.tool_defs + mcp.tool_defs
                trace = TraceWriter(
                    cwd=str(dirs.workspace), model=slug, tools=[t.name for t in tool_defs],
                    mcp_servers=[{"name": f"plugin:{c.plugin}:{c.name}", "status": "connected"
                                  if any(t.server == c.name for t in mcp.tool_defs) else "failed", "source": "plugin"}
                                 for c in servers],
                    skills=[sk.qualified_name for sk in skills],
                    plugins=[{"name": p.name, "path": str(p.root)} for p in plugins] if with_plugin else [])
                trace_holder["t"] = trace
                for n in notices:
                    trace.system("skilleval_notice", notice=n)
                deferred = deferred_tool_names(s, tool_defs)
                system_prompt = build_system_prompt(cwd=str(dirs.workspace), skills=skills, tool_defs=tool_defs,
                                                    mcp_instructions=mcp.instructions,
                                                    append=case.append_system_prompt, today=date.today(), model=slug,
                                                    deferred_tools=deferred)
                history = load_history(case.dir / case.context.history_file) if case.context.history_file else None
                outcome = await run_agent(settings=s, model_slug=slug, system_prompt=system_prompt,
                                          prompt=case.prompt, providers=[builtin, mcp], trace=trace,
                                          max_turns=case.max_turns, timeout_seconds=case.timeout_seconds,
                                          history=history, simulated_user=sim, judge=self.judge,
                                          model=self.model_factory(case, arm, idx) if self.model_factory else None)
                mock_calls = list(mcp.calls)
                for n in mcp.notices:
                    if n not in notices:
                        notices.append(n)
                if outcome.error or outcome.aborted:
                    mcp.discard_recordings()

            trace_path = out_dir / "traces" / _safe(case.name) / f"{arm}-{idx}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(s.mask(trace.text()), encoding="utf-8")

            art = RunArtifacts(final_message=outcome.final_message, trace_lines=[s.mask(x) for x in trace.lines],
                               tool_calls=outcome.tool_calls, files_created=builtin.files_created(),
                               workspace=dirs.workspace, mock_calls=mock_calls, turns=outcome.turns,
                               error=outcome.error, aborted=outcome.aborted, api_error=outcome.api_error,
                               cost_usd=outcome.cost_usd, duration_seconds=time.monotonic() - t0,
                               trace_path=str(trace_path))
            skip_paid = o.max_cost_usd is not None and self.spent + outcome.cost_usd >= o.max_cost_usd
            graders, score, judge_cost = await grade_run(case, art, self.judge, excluded=excluded,
                                                         skip_paid=skip_paid)
            if dirs.keep:
                notices.append(f"workspace kept: {dirs.root}")
            return RunResult(score=score, passed=(not outcome.api_error) and score >= o.threshold,
                             turns=outcome.turns, cost_usd=outcome.cost_usd, judge_cost_usd=judge_cost,
                             duration_seconds=time.monotonic() - t0, started_at=started_iso,
                             error=outcome.error, trace_path=str(trace_path),
                             skipped_paid_graders=skip_paid and any(g.type in ("llm", "baseline")
                                                                    for g in case.graders),
                             aborted=outcome.aborted, graders=graders, notices=notices,
                             api_error=outcome.api_error)
        except ConfigError as ex:
            return fail(f"config error: {ex}")
        finally:
            if ext and ext.teardown and o.hooks:
                if err := await sandbox.run_hook(ext.teardown, case=case, env=hook_env):
                    notices.append(f"teardown failed: {err}")
            sandbox.cleanup(dirs)

    # --- sada ---------------------------------------------------------------------------------------------------------

    async def run(self) -> int:
        o = self.opts
        try:
            suite = self.prepare()
        except (ValueError, ConfigError, OSError) as ex:
            self.warn(f"Error: {ex}")
            return 1
        assert self.settings is not None
        for le in suite.load_errors:
            self.warn(f"Error: {le.path}: {le.message}")
        if not suite.cases:
            self.warn("No eval cases found" + (" (filters matched nothing)" if o.case_globs or o.tags else ""))
            return 1
        if not self.settings.api_key and self.model_factory is None:
            self.warn("Error: chybí OPENROUTER_API_KEY (env, --env-file nebo env_files v skilleval.toml)")
            return 1

        self.results_dir = Path(o.output_dir).resolve() if o.output_dir else default_results_dir(
            suite.eval_dir, self.started)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        jobs: list[tuple[CaseSpec, str, int, int, list[PluginInfo], set[str]]] = []
        for case in suite.cases:
            try:
                plugins = self.case_plugins(case)
            except ConfigError as ex:
                self.warn(f"Error: {case.name}: {ex}")
                self.exit_code = 1
                continue
            arms, reason = self.arms_for(case, plugins)
            n = o.runs or case.runs or 3
            cr = CaseResult(case=case, runs_per_case=n, without_runs=[] if "without" in arms else None,
                            single_arm_reason=reason)
            self.results[case.name] = cr
            if reason and o.ablation != "none":
                self.warn(f"single-arm (no Δ): {case.name} – {reason}")
            plugin_servers = {name for p in plugins for name in p.mcp_servers}
            excluded = excluded_graders(case, two_arm=len(arms) == 2, plugin_servers=plugin_servers)
            for arm in arms:
                for i in range(1, n + 1):
                    jobs.append((case, arm, i, n, plugins, excluded))

        sem = asyncio.Semaphore(max(1, min(8, o.concurrency)))
        stop = asyncio.Event()

        async def worker(job: tuple[CaseSpec, str, int, int, list[PluginInfo], set[str]]) -> None:
            case, arm, i, n, plugins, excluded = job
            async with sem:
                if stop.is_set():
                    return
                if o.max_cost_usd is not None and self.spent >= o.max_cost_usd:
                    self.partial_reason = self.partial_reason or "cost_ceiling"
                    return
                self.runs_started += 1
                rr = await self.execute_run(case, arm, i, n, plugins, excluded, self.results_dir)
                self.spent += rr.cost_usd + rr.judge_cost_usd
                cr = self.results[case.name]
                (cr.with_runs if arm == "with" else cr.without_runs).append(rr)  # type: ignore[union-attr]
                if rr.api_error:
                    self.api_errors += 1
                    if self.runs_ok == 0 and _is_auth_error(rr.error or ""):
                        self.partial_reason = "auth_failed"
                        stop.set()
                else:
                    self.runs_ok += 1
                self.log(progress_line(case.name, arm, i, n, rr, color=self.color)
                         + "".join(f"\n    {x}" for x in rr.notices if o.verbose or not x.startswith("workspace")))

        tasks = [asyncio.create_task(worker(j)) for j in jobs]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            self.partial_reason = "interrupted"
            self.finalize()
            raise
        return self.finalize()

    def finalize(self) -> int:
        o = self.opts
        assert self.suite is not None and self.settings is not None and self.results_dir is not None
        crs = [cr for cr in self.results.values()]
        for cr in crs:  # pořadí runů dle startu (souběh)
            cr.with_runs.sort(key=lambda r: r.started_at)
            if cr.without_runs:
                cr.without_runs.sort(key=lambda r: r.started_at)
        abl = o.ablation or ("with-without" if any(cr.without_runs is not None for cr in crs) else "none")
        agg = build_aggregate(
            suite=self.suite, case_results=crs, started_at=self.started_iso,
            duration_seconds=time.monotonic() - self.t0, threshold=o.threshold, ablation=abl,
            concurrency=o.concurrency, model_override=o.model, case_filter=",".join(o.case_globs) or None,
            judge_model=self.settings.resolve_model(o.judge_model, judge=True),
            agent_model=self.settings.resolve_model(o.model), partial=self.partial_reason is not None,
            partial_reason=self.partial_reason, runner_version=__version__)
        write_results(self.results_dir, agg)
        report_path = None
        if not o.no_report:
            try:
                report_path = render_report(agg, self.results_dir)
            except Exception as ex:  # report nesmí shodit výsledek
                self.warn(f"Warning: report.html se nepodařilo vytvořit: {ex}")
        try:
            write_adopt(self.results_dir)
        except Exception as ex:  # pragma: no cover
            self.warn(f"Warning: ADOPT.txt: {ex}")

        if o.json_out is not None:
            text = json.dumps(agg, ensure_ascii=False, indent=2)
            if o.json_out == "-":
                print(text, file=self.out)
            else:
                Path(o.json_out).write_text(text, encoding="utf-8")
        else:
            print(summary_table(agg, report_path=report_path or self.results_dir / "aggregate-result.json"),
                  file=self.out)

        if self.api_errors:
            self.warn(f"{self.api_errors} run(s) ended on an API error and scored 0 without grading")
        if self.partial_reason == "interrupted":
            return 130
        if self.partial_reason in ("cost_ceiling", "auth_failed"):
            return 2
        failed = [cr.case.name for cr in crs if not case_passed(cr, o.threshold)]
        if failed or self.api_errors or self.suite.load_errors or self.exit_code:
            return 1
        return 0


def _is_auth_error(error: str) -> bool:
    e = error.lower()
    return "401" in e or "authentication" in e or "invalid api key" in e or "no auth" in e


def run_suite(opts: RunOptions, *, out: TextIO | None = None, err: TextIO | None = None, **kw: Any) -> int:
    runner = Runner(opts, out=out, err=err, **kw)
    try:
        return asyncio.run(runner.run())
    except KeyboardInterrupt:
        return 130
