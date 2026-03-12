"""
Agent 3 — Pytest Runner & Feedback Generator
"""

from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
from llms import call_groq_analyst
from prompts import TEST_FEEDBACK_SYSTEM_PROMPT, build_feedback_user_prompt


logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"

REPORT_FILE = Path(tempfile.gettempdir()) / "pytest_report.json"
TEMP_DIR = Path(tempfile.gettempdir()) / "company_agent_tests"
ERROR_REPORT_DIR = TEMP_DIR / "error_reports"


# ─────────────────────────────────────────
# TEMP DIRECTORY SETUP
# ─────────────────────────────────────────

def _prepare_temp_dirs() -> None:

    TEMP_DIR.mkdir(exist_ok=True)
    ERROR_REPORT_DIR.mkdir(exist_ok=True)

    for old in ERROR_REPORT_DIR.glob("*.csv"):
        old.unlink(missing_ok=True)

    logger.debug("[Agent3] Cleared old CSV error reports")


# ─────────────────────────────────────────
# WRITE COMPANY JSON
# ─────────────────────────────────────────

def _write_company_json(data: dict) -> Path:

    path = Path(tempfile.gettempdir()) / "current_company.json"
    path.write_text(json.dumps(data, indent=2))

    return path


# ─────────────────────────────────────────
# RUN PYTEST
# ─────────────────────────────────────────

def _run_pytest(company_json_path: Path) -> dict:

    if REPORT_FILE.exists():
        REPORT_FILE.unlink()



    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TESTS_DIR),
        f"--company-json={company_json_path}",
        "--json-report",
        f"--json-report-file={REPORT_FILE}",
        "--tb=short",
        "-q",
        "-s",
    ]
    logger.info("[Agent3] Running pytest command:")
    logger.info(" ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    logger.info("[Agent3] pytest stdout:\n%s", result.stdout)
    logger.info("[Agent3] pytest stderr:\n%s", result.stderr)

    if REPORT_FILE.exists():
        try:
            return json.loads(REPORT_FILE.read_text())
        except Exception as e:
            logger.error("[Agent3] Failed to parse pytest JSON report: %s", e)

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "tests": [],
        "summary": {},
    }


# ─────────────────────────────────────────
# EXTRACT TEST SUMMARY
# ─────────────────────────────────────────

def _extract_test_summary(report: dict) -> dict:

    summary = report.get("summary", {})

    return {
        "total": summary.get("total", 0),
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "errors": summary.get("errors", 0),
        "skipped": summary.get("skipped", 0),
    }


# ─────────────────────────────────────────
# PARSE PYTEST FAILURES
# ─────────────────────────────────────────

def _parse_pytest_failures(report: dict, company_data: dict) -> List[Dict[str, Any]]:
    """
    Parse pytest failures. These are general test failures.
    Field-level errors will come from CSV reports.
    """

    failures = []

    for test in report.get("tests", []):

        if test.get("outcome") not in ("failed", "error"):
            continue

        node_id = test.get("nodeid", "")

        call_info = test.get("call", {})
        longrepr = call_info.get("longrepr", "")

        reason = longrepr[:500] if longrepr else "pytest failure"

        failures.append(
            {
                "field": "general_validation",
                "test": node_id,
                "reason": reason,
                "value": "N/A",
                "source": "pytest",
            }
        )

    return failures

# ─────────────────────────────────────────
# PARSE CSV ERROR REPORTS
# ─────────────────────────────────────────

def _parse_csv_error_reports(company_data: dict) -> List[Dict[str, Any]]:
    
    import time
    time.sleep(0.5)

    failures = []

    for csv_file in ERROR_REPORT_DIR.glob("*.csv"):

        logger.info("[Agent3] Reading CSV report: %s", csv_file.name)

        try:

            with open(csv_file, newline="", encoding="utf-8") as f:

                reader = csv.DictReader(f)

                for row in reader:

                    field = row.get("column_name", "unknown")

                    failures.append(
                        {
                            "field": field,
                            "test": row.get("test_case_id", csv_file.stem),
                            "reason": row.get("error_message", "validation failed"),
                            "value": row.get("input_value", company_data.get(field, "N/A")),
                            "source": f"csv_report:{csv_file.name}",
                        }
                    )

        except Exception as exc:

            logger.warning("[Agent3] Failed parsing CSV %s: %s", csv_file, exc)

    return failures

# ─────────────────────────────────────────
# MERGE FAILURES
# ─────────────────────────────────────────

def _merge_failures(pytest_failures: List[Dict], csv_failures: List[Dict]):

    seen = set()
    merged = []

    for f in pytest_failures + csv_failures:

        key = (f["field"], f["test"])

        if key not in seen:
            seen.add(key)
            merged.append(f)

    return merged


# ─────────────────────────────────────────
# GENERATE LLM FEEDBACK
# ─────────────────────────────────────────

def _generate_feedback(company_data: dict, failures: List[Dict]):

    if not failures:
        return ""

    user_prompt = build_feedback_user_prompt(
        json.dumps(company_data, indent=2),
        failures,
    )

    raw = call_groq_analyst(TEST_FEEDBACK_SYSTEM_PROMPT, user_prompt)

    try:

        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)

        text = m.group(1) if m else raw.strip()

        parsed = json.loads(text)

        corrections = parsed.get("corrections", [])

        lines = []

        for c in corrections:

            lines.append(
                f"FIELD: {c['field']}\n"
                f"ISSUE: {c['issue']}\n"
                f"FIX: {c['fix']}\n"
                f"EXAMPLE: {c.get('example','N/A')}"
            )

        return "\n\n".join(lines)

    except Exception:

        return raw


# ─────────────────────────────────────────
# MAIN AGENT ENTRY
# ─────────────────────────────────────────

def run_agent3(
    company_data: dict,
) -> Tuple[bool, str, List[Dict[str, Any]], Dict[str, int]]:

    _prepare_temp_dirs()

    json_path = _write_company_json(company_data)

    logger.info("[Agent3] Company JSON written to %s", json_path)

    report = _run_pytest(json_path)

    test_summary = _extract_test_summary(report)

    logger.info(
        "[Agent3] Tests run: %d | Passed: %d | Failed: %d",
        test_summary["total"],
        test_summary["passed"],
        test_summary["failed"],
    )

    if test_summary["total"] == 0:
        logger.error("⚠️ Pytest discovered ZERO tests! Something is wrong with test discovery.")
        logger.error("STDOUT:\n%s", report.get("stdout"))
        logger.error("STDERR:\n%s", report.get("stderr"))

    pytest_failures = _parse_pytest_failures(report, company_data)

    csv_failures = _parse_csv_error_reports(company_data)

    all_failures = _merge_failures(pytest_failures, csv_failures)

    logger.info("[Agent3] Total failures detected: %d", len(all_failures))

    all_passed = len(all_failures) == 0

    if all_passed:

        logger.info("[Agent3] ✅ All tests passed")

        return True, "", [], test_summary

    logger.info("[Agent3] Generating correction feedback via Groq")
    fields_to_fix = list({f["field"] for f in all_failures if f["field"] != "general_validation"})
    feedback = _generate_feedback(company_data, all_failures)

    return False, feedback, all_failures, test_summary