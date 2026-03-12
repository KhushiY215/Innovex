#!/usr/bin/env python3
# main.py
"""
Company Intelligence Agent — Main Entry Point
─────────────────────────────────────────────
Usage:
    python main.py "Apple Inc"
    python main.py "Apple Inc" --max-iterations 3
    python main.py "Apple Inc" --output-dir results/

Output files produced (per iteration + final):
  outputs/<company>_iter<N>_llm_outputs.json   ← Agent 1: raw output from all 3 LLMs
  outputs/<company>_iter<N>_consolidated.json  ← Agent 2: best consolidated JSON
  outputs/<company>_iter<N>_test_report.json   ← Agent 3: failed test cases + fixes
  outputs/<company>.json                        ← Final validated output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

import logging
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich         import box

from config.settings import settings
from graph.workflow  import run_pipeline

# ─── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

console = Console()


# ─── cli ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Company Intelligence Agent — extract 163 parameters for any company",
    )
    p.add_argument("company_name", help="Name of the company (e.g. 'Apple Inc')")
    p.add_argument(
        "--max-iterations", type=int, default=settings.max_iterations,
        help=f"Max Agent1→Agent2→Agent3 loop iterations (default: {settings.max_iterations})",
    )
    p.add_argument(
        "--output-dir", default=settings.output_dir,
        help=f"Directory for all output JSON files (default: {settings.output_dir})",
    )
    return p.parse_args()


# ─── output file scanner ──────────────────────────────────────────────────────

def _collect_outputs(out_dir: Path, safe_name: str, iterations_run: int) -> dict:
    """
    Collect all output file paths produced during the pipeline run,
    grouped by type.
    """
    files = {
        "agent1_llm_outputs":  [],
        "agent2_consolidated": [],
        "agent3_test_reports": [],
        "final_output":        None,
    }

    for i in range(1, iterations_run + 1):
        p1 = out_dir / f"{safe_name}_iter{i}_llm_outputs.json"
        p2 = out_dir / f"{safe_name}_iter{i}_consolidated.json"
        p3 = out_dir / f"{safe_name}_iter{i}_test_report.json"
        if p1.exists(): files["agent1_llm_outputs"].append(str(p1))
        if p2.exists(): files["agent2_consolidated"].append(str(p2))
        if p3.exists(): files["agent3_test_reports"].append(str(p3))

    final = out_dir / f"{safe_name}.json"
    if final.exists():
        files["final_output"] = str(final)

    return files


# ─── display helpers ──────────────────────────────────────────────────────────

def _print_output_table(files: dict, iterations: int) -> None:
    table = Table(
        title="📁  All Output Files",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Step",        style="bold white",  min_width=20)
    table.add_column("Iteration",   style="cyan",        justify="center")
    table.add_column("File",        style="yellow")
    table.add_column("Description", style="dim white",   min_width=40)

    for i, path in enumerate(files["agent1_llm_outputs"], 1):
        table.add_row(
            "🔵 Agent 1 — LLM Outputs", str(i), Path(path).name,
            "Raw output from all 3 LLMs before consolidation",
        )

    for i, path in enumerate(files["agent2_consolidated"], 1):
        table.add_row(
            "🟡 Agent 2 — Consolidated", str(i), Path(path).name,
            "Best single JSON produced by the judge LLM",
        )

    for i, path in enumerate(files["agent3_test_reports"], 1):
        table.add_row(
            "🔴 Agent 3 — Test Report", str(i), Path(path).name,
            "Failed pytest cases + LLM correction instructions",
        )

    if files["final_output"]:
        table.add_row(
            "✅  Final Output", "—", Path(files["final_output"]).name,
            "Validated consolidated JSON (all 163 parameters)",
        )

    console.print(table)


def _print_final_summary(data: dict, files: dict, iterations: int, passed: bool) -> None:
    """Print a quick-look summary of the final company data."""
    name      = data.get("name", "N/A")
    short     = data.get("short_name", "")
    hq        = data.get("headquarters_address", "N/A")
    ceo       = data.get("ceo_name", "N/A")
    emp       = data.get("employee_size", "N/A")
    rev       = data.get("annual_revenue")
    maturity  = data.get("company_maturity", "N/A")
    website   = data.get("website_url", "N/A")

    rev_str   = f"${rev:,.0f}" if isinstance(rev, (int, float)) else "N/A"

    summary = (
        f"[bold cyan]{name}[/bold cyan]"
        + (f"  ([dim]{short}[/dim])" if short else "") + "\n"
        f"[dim]HQ:[/dim]          {hq}\n"
        f"[dim]CEO:[/dim]         {ceo}\n"
        f"[dim]Employees:[/dim]   {emp}\n"
        f"[dim]Revenue:[/dim]     {rev_str}\n"
        f"[dim]Maturity:[/dim]    {maturity}\n"
        f"[dim]Website:[/dim]     {website}\n\n"
        f"[dim]Iterations run:[/dim]  {iterations}\n"
        f"[dim]Tests passed:[/dim]    {'[bold green]Yes ✅[/bold green]' if passed else '[yellow]Best effort 🟡[/yellow]'}"
    )

    console.print(Panel(summary, title="Company Summary", border_style="green"))


def _print_test_failures(test_report_path: str) -> None:
    """
    Display detailed pytest results including number of tests run,
    passed, failed, and failed fields.
    """
    try:
        report = json.loads(Path(test_report_path).read_text())

        summary = report.get("summary", {})
        failures = report.get("failures", [])

        total_tests = summary.get("total", "N/A")
        passed      = summary.get("passed", "N/A")
        failed      = summary.get("failed", len(failures))

        pass_rate = "N/A"
        if isinstance(total_tests, int) and total_tests > 0 and isinstance(passed, int):
            pass_rate = f"{(passed / total_tests) * 100:.1f}%"

        # ── Summary panel ─────────────────────────────
        summary_text = (
            f"[bold cyan]Total tests run:[/bold cyan] {total_tests}\n"
            f"[bold green]Passed:[/bold green] {passed}\n"
            f"[bold red]Failed:[/bold red] {failed}\n"
            f"[bold yellow]Pass rate:[/bold yellow] {pass_rate}"
        )

        console.print(
            Panel(summary_text, title="🧪 Test Results", border_style="cyan")
        )

        # ── Failure table ─────────────────────────────
        if failures:
            table = Table(
                title=f"⚠️ Failed Field Validations ({len(failures)})",
                box=box.SIMPLE_HEAD,
                title_style="bold red",
            )

            table.add_column("Field", style="red bold")
            table.add_column("Current Value", style="yellow", max_width=30)
            table.add_column("Failure Reason", style="white", max_width=60)

            for f in failures:
                table.add_row(
                    f.get("field", "?"),
                    str(f.get("current_value", "N/A"))[:80],
                    str(f.get("failure_reason", ""))[:120],
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error reading test report:[/red] {e}")
@traceable(name="company-intelligence-agent")
def traced_pipeline(company_name: str, max_iterations: int):
    return run_pipeline(
        company_name=company_name,
        max_iterations=max_iterations,
    )
# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    settings.output_dir = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    safe_name = args.company_name.replace(" ", "_").replace("/", "-")

    console.print(
        Panel(
            Text(f"🏢  Researching: {args.company_name}", style="bold cyan"),
            subtitle=f"Max iterations: {args.max_iterations}  |  Output dir: {args.output_dir}",
        )
    )

    # ── Run pipeline ──────────────────────────────────────────────────────
    final_state = traced_pipeline(
    company_name=args.company_name,
    max_iterations=args.max_iterations,
)

    iterations   = final_state.get("iteration", 0)
    passed       = final_state.get("test_passed", False)
    output_path  = final_state.get("output_path")

    if not output_path:
        console.print("[bold red]Pipeline did not produce a final output file.[/bold red]")
        sys.exit(1)

    # ── Collect all files produced ────────────────────────────────────────
    out_dir = Path(args.output_dir)
    files   = _collect_outputs(out_dir, safe_name, iterations)

    console.print()

    # 1. Output file table
    _print_output_table(files, iterations)
    console.print()

    # 2. Company summary panel
    final_data = json.loads(Path(output_path).read_text())
    _print_final_summary(final_data, files, iterations, passed)
    console.print()

    # 3. Last test report failures (if any)
    if files["agent3_test_reports"]:
        _print_test_failures(files["agent3_test_reports"][-1])

    # 4. Plain file list for easy copy-paste
    console.print("[bold]Output files:[/bold]")
    for path in files["agent1_llm_outputs"]:
        console.print(f"  [blue]Agent 1 →[/blue] {path}")
    for path in files["agent2_consolidated"]:
        console.print(f"  [yellow]Agent 2 →[/yellow] {path}")
    for path in files["agent3_test_reports"]:
        console.print(f"  [red]Agent 3 →[/red] {path}")
    if files["final_output"]:
        console.print(f"  [green]Final   →[/green] {files['final_output']}")
    console.print()


if __name__ == "__main__":
    main()