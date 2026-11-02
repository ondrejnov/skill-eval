"""CLI `skilleval`."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from skilleval import __version__
from skilleval.config import ConfigError, load_settings
from skilleval.runner import RunOptions, run_suite


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="skilleval.toml (jinak CWD, kořen pluginu, adresář aplikace)")
    p.add_argument("--env-file", action="append", default=[], dest="env_files", help="další .env (opakovatelně)")
    p.add_argument("--eval-dir", help="eval adresář relativně k pluginu (default evals/ nebo experimental.evals)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="skilleval", description="Evaly skillů/pluginů (formát Claude Code plugin evals) "
                                                              "přes OpenRouter.")
    ap.add_argument("--version", action="version", version=f"skilleval {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="spustí sadu")
    r.add_argument("target", nargs="?", help="adresář pluginu, prompt.md/case.yaml (default CWD)")
    _add_common(r)
    r.add_argument("--case", action="append", default=[], dest="case_globs", help="glob názvu case (opakovatelně)")
    r.add_argument("--tag", action="append", default=[], dest="tags", help="tag (opakovatelně, any)")
    r.add_argument("--plugin", action="append", default=[], dest="extra_plugins",
                   help="další plugin/skill do with-armu (opakovatelně)")
    r.add_argument("--runs", type=int, help="runů na case a arm (default case.runs, jinak 3)")
    r.add_argument("-j", "--concurrency", type=int, default=1, help="souběžných runů (1–8)")
    r.add_argument("--model", help="model agenta (alias haiku/sonnet/opus nebo OpenRouter slug)")
    r.add_argument("--judge-model", help="model soudce (default [models].judge)")
    r.add_argument("--ablation", choices=["none", "with-without"], help="none = jen with-arm")
    r.add_argument("--threshold", type=float, default=1.0)
    r.add_argument("--max-cost-usd", type=float)
    r.add_argument("--allow-tools", nargs="+", default=[], help="granty toolů (Write Edit 'mcp__plugin_x_y__*')")
    r.add_argument("--strict-grants", action="store_true", help="MCP tooly reálných serverů jen s --allow-tools (jako CC)")
    r.add_argument("--mocks", choices=["record", "off"], default="record")
    r.add_argument("--allow-real-servers", action="store_true")
    r.add_argument("--scaffold", action="store_true", help="spouštět context.scaffold_script")
    r.add_argument("--hooks", action="store_true", help="spouštět setup/teardown/fault run ze skilleval.yaml")
    r.add_argument("--defer-tools", choices=["auto", "on", "off"], help="odkládat MCP tooly za ToolSearch (default auto)")
    r.add_argument("--mcp-config", help="JSON {mcpServers: {...}} – přepis serverů pluginu")
    r.add_argument("--mcp-profile", help="profil z [mcp.profiles] pro všechny cases")
    r.add_argument("--json", nargs="?", const="-", dest="json_out", help="výsledek jako JSON na stdout / do souboru .json")
    r.add_argument("--output-dir")
    r.add_argument("--keep-temp", action="store_true")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--no-color", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true")

    for name, hlp in (("list", "vypíše cases"), ("validate", "zvaliduje sadu")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("target", nargs="?")
        _add_common(p)
        p.add_argument("--case", action="append", default=[], dest="case_globs")
        p.add_argument("--tag", action="append", default=[], dest="tags")

    p = sub.add_parser("report", help="znovu vygeneruje report.html z výsledků")
    p.add_argument("results", help="adresář výsledků nebo aggregate-result.json")
    p.add_argument("--output", help="cílový soubor (default report.html; u výsledků z Claude Code report-skilleval.html)")

    p = sub.add_parser("compare", help="porovná dva výsledky")
    p.add_argument("a")
    p.add_argument("b")

    p = sub.add_parser("models", help="ověří aliasy modelů proti OpenRouteru")
    p.add_argument("--config")
    p.add_argument("--env-file", action="append", default=[], dest="env_files")

    p = sub.add_parser("mocks", help="práce s mocky")
    msub = p.add_subparsers(dest="mocks_cmd", required=True)
    s = msub.add_parser("snapshot", help="uloží _tools.json z reálného serveru")
    s.add_argument("server")
    s.add_argument("target", nargs="?")
    _add_common(s)
    s.add_argument("--mcp-profile")
    s.add_argument("--out", help="cílový adresář (default <eval>/mocks/<server>)")

    p = sub.add_parser("init", help="šablona case")
    p.add_argument("--bare", required=True, metavar="NAME")
    p.add_argument("target", nargs="?")
    p.add_argument("--eval-dir")
    return ap


def cmd_run(a: argparse.Namespace) -> int:
    if a.json_out not in (None, "-") and not a.json_out.endswith(".json"):
        print("Error: --json output path must end in .json", file=sys.stderr)
        return 1
    opts = RunOptions(
        target=a.target, case_globs=a.case_globs, tags=a.tags, eval_dir=a.eval_dir, extra_plugins=a.extra_plugins,
        runs=a.runs, concurrency=a.concurrency, model=a.model, judge_model=a.judge_model, ablation=a.ablation,
        threshold=a.threshold, max_cost_usd=a.max_cost_usd, allow_tools=a.allow_tools, strict_grants=a.strict_grants,
        mocks=a.mocks, allow_real_servers=a.allow_real_servers, scaffold=a.scaffold, hooks=a.hooks,
        mcp_config=a.mcp_config, mcp_profile=a.mcp_profile, config=a.config, env_files=a.env_files,
        json_out=a.json_out, output_dir=a.output_dir, keep_temp=a.keep_temp, verbose=a.verbose,
        no_report=a.no_report, color=False if a.no_color else None, defer_tools=a.defer_tools)
    if not (1 <= opts.concurrency <= 8):
        print("Error: --concurrency musí být 1–8", file=sys.stderr)
        return 1
    if not (0 <= opts.threshold <= 1):
        print("Error: --threshold musí být 0..1", file=sys.stderr)
        return 1
    return run_suite(opts)


def _load(a: argparse.Namespace):
    from skilleval.plugin import find_plugin_root
    from skilleval.suite.loader import load_suite

    t = Path(a.target).resolve() if a.target else Path.cwd()
    settings = load_settings(a.config, a.env_files, plugin_root=find_plugin_root(t if t.is_dir() else t.parent))
    suite = load_suite(a.target, settings, eval_dir=a.eval_dir, case_globs=a.case_globs or None, tags=a.tags or None,
                       resolve_mcp=False)
    return settings, suite


def cmd_validate(a: argparse.Namespace) -> int:
    from skilleval.suite.validate import validate_suite

    settings, suite = _load(a)
    errors = validate_suite(suite, settings)
    for e in errors:
        print(f"Error: {e.path}: {e.message}")
    print(f"{len(suite.cases)} case(s), {len(errors)} error(s)")
    return 1 if errors or not suite.cases else 0


def cmd_list(a: argparse.Namespace) -> int:
    settings, suite = _load(a)
    p = suite.plugin
    if p:
        print(f"plugin {p.name} ({p.root}) – skills: {', '.join(s.name for s in p.skills) or '—'}; "
              f"MCP: {', '.join(p.mcp_servers) or '—'}")
    print(f"eval dir {suite.eval_dir}")
    for c in suite.cases:
        graders = ", ".join(f"{g.name}:{g.type}" for g in c.graders)
        extra = []
        if c.mock_dirs:
            extra.append("mocks")
        if c.extension:
            ext = c.extension
            extra += [k for k, v in (("profile", ext.mcp_profile), ("faults", ext.faults), ("setup", ext.setup),
                                     ("teardown", ext.teardown), ("simulated_user", ext.simulated_user)) if v]
        print(f"- {c.name}  [{', '.join(c.all_tags)}]  turns {c.max_turns} · {c.timeout_seconds}s"
              f"{' · tools ' + ','.join(c.allowed_tools) if c.allowed_tools else ''}"
              f"{' · ' + ', '.join(extra) if extra else ''}\n    graders: {graders}")
    for e in suite.load_errors:
        print(f"Error: {e.path}: {e.message}")
    return 1 if suite.load_errors else 0


def cmd_report(a: argparse.Namespace) -> int:
    from skilleval.results import load_aggregate
    from skilleval.results.report import REPORT_FILE, render_html

    path = Path(a.results)
    agg = load_aggregate(path)
    out_dir = path if path.is_dir() else path.parent
    if a.output:
        target = Path(a.output)
    elif "runner" in agg:
        target = out_dir / REPORT_FILE
    else:  # výsledek jiného nástroje (Claude Code): jeho report.html nikdy nepřepisovat
        target = out_dir / "report-skilleval.html"
    target.write_text(render_html(agg, out_dir), encoding="utf-8")
    print(target)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    from skilleval.results import compare, load_aggregate

    print(compare(load_aggregate(Path(a.a)), load_aggregate(Path(a.b))))
    return 0


def cmd_models(a: argparse.Namespace) -> int:
    import httpx

    s = load_settings(a.config, a.env_files)
    resp = httpx.get(f"{s.base_url}/models", timeout=30)
    resp.raise_for_status()
    models = {m["id"]: m for m in resp.json().get("data", [])}
    rc = 0
    for alias, slug in sorted(s.aliases.items()):
        m = models.get(slug)
        if m:
            pr = m.get("pricing", {})
            print(f"{alias:10} {slug:40} ok   in ${float(pr.get('prompt', 0)) * 1e6:.2f}/M  "
                  f"out ${float(pr.get('completion', 0)) * 1e6:.2f}/M")
        else:
            print(f"{alias:10} {slug:40} NOT FOUND")
            rc = 1
    print(f"default {s.resolve_model(None)} · judge {s.resolve_model(None, judge=True)}")
    return rc


def cmd_mocks_snapshot(a: argparse.Namespace) -> int:
    from skilleval.mcp.client import RealServer
    from skilleval.plugin import find_plugin_root, load_plugin

    t = Path(a.target).resolve() if a.target else Path.cwd()
    root = find_plugin_root(t)
    if not root:
        print(f"Error: {t} není plugin", file=sys.stderr)
        return 1
    settings = load_settings(a.config, a.env_files, plugin_root=root)
    plugin = load_plugin(root, settings, profile=a.mcp_profile, eval_dir=a.eval_dir)
    cfg = plugin.mcp_servers.get(a.server)
    if not cfg:
        print(f"Error: plugin {plugin.name} nemá MCP server {a.server}", file=sys.stderr)
        return 1

    async def go() -> list[dict]:
        srv = RealServer(cfg, env={}, connect_timeout=30.0, call_timeout=60.0)
        await srv.start()
        try:
            return [t.model_dump(mode="json", exclude_none=True, by_alias=True) for t in srv.tools]
        finally:
            await srv.close()

    tools = asyncio.run(go())
    out = Path(a.out) if a.out else (plugin.eval_dir or root / "evals") / "mocks" / a.server
    out.mkdir(parents=True, exist_ok=True)
    (out / "_tools.json").write_text(json.dumps({"tools": tools}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(tools)} tools → {out / '_tools.json'}")
    return 0


PROMPT_TEMPLATE = """---
max_turns: 10
---

<Napiš požadavek, jak by ho napsal uživatel – bez názvu skillu.>
"""

GRADER_TEMPLATE = """---
type: llm
---

PASS if <co obsahuje správná odpověď>.
FAIL if <jak vypadá špatná nebo chybějící odpověď>.
"""


def cmd_init(a: argparse.Namespace) -> int:
    from skilleval.plugin import find_plugin_root, plugin_eval_dir

    t = Path(a.target).resolve() if a.target else Path.cwd()
    root = find_plugin_root(t) or t
    edir, _ = plugin_eval_dir(root, a.eval_dir)
    case = edir / a.bare
    if case.exists():
        print(f"Error: {case} už existuje", file=sys.stderr)
        return 1
    (case / "graders").mkdir(parents=True)
    (case / "prompt.md").write_text(PROMPT_TEMPLATE, encoding="utf-8")
    (case / "graders" / "criteria.md").write_text(GRADER_TEMPLATE, encoding="utf-8")
    print(f"{case}/prompt.md\n{case}/graders/criteria.md")
    return 0


def main(argv: list[str] | None = None) -> None:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    a = build_parser().parse_args(argv)
    handlers = {"run": cmd_run, "validate": cmd_validate, "list": cmd_list, "report": cmd_report,
                "compare": cmd_compare, "models": cmd_models, "init": cmd_init}
    try:
        if a.cmd == "mocks":
            rc = cmd_mocks_snapshot(a)
        else:
            rc = handlers[a.cmd](a)
    except (ConfigError, ValueError, FileNotFoundError) as ex:
        print(f"Error: {ex}", file=sys.stderr)
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
