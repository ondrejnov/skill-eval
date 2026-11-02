"""Výsledky běhu: aggregate-result.json (aggregate), report.html (report), terminál (summary), compare."""

from skilleval.results.aggregate import (
    build_aggregate,
    case_aggregates,
    case_passed,
    default_results_dir,
    load_aggregate,
    write_results,
)
from skilleval.results.compare import compare
from skilleval.results.report import render_report
from skilleval.results.summary import progress_line, summary_table

__all__ = ["build_aggregate", "case_aggregates", "case_passed", "compare", "default_results_dir", "load_aggregate",
           "progress_line", "render_report", "summary_table", "write_results"]
