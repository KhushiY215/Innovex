# tests/conftest.py
"""
Shared pytest fixtures.

Fixtures provided
─────────────────
  company      → dict   loaded from --company-json (or temp JSON)
  csv_path     → Path   staging_company_rows.csv written from company JSON
  error_report_dir → Path  temp directory where test CSV error reports are saved
"""

import csv
import json
import tempfile
import pytest
from pathlib import Path


# ─── CLI option ───────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--company-json",
        action="store",
        default=None,
        help="Path to the company JSON file under test",
    )


# ─── temp directory (shared across session) ───────────────────────────────────

_TEMP_DIR = Path(tempfile.gettempdir()) / "company_agent_tests"
_TEMP_DIR.mkdir(exist_ok=True)


# ─── company JSON fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def company(request) -> dict:
    """Load company data dict from --company-json or default temp path."""
    path_str = request.config.getoption("--company-json")
    if path_str:
        path = Path(path_str)
    else:
        path = Path(tempfile.gettempdir()) / "current_company.json"

    if not path.exists():
        pytest.skip(f"Company JSON not found at {path}")

    return json.loads(path.read_text())


# ─── CSV fixture (used by TC-1.1 company-name tests) ─────────────────────────

@pytest.fixture(scope="session")
def csv_path(company) -> Path:
    """
    Write the company dict as a single-row CSV (staging_company_rows.csv)
    so CSV-based test cases can read it with pd.read_csv.

    All 163 fields become columns; the single row is the current company.
    """
    path = _TEMP_DIR / "staging_company_rows.csv"

    # Flatten nested fields to JSON strings for CSV compatibility
    flat = {}
    for k, v in company.items():
        if isinstance(v, (dict, list)):
            flat[k] = json.dumps(v)
        else:
            flat[k] = v if v is not None else ""

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    return path


# ─── error report directory fixture ──────────────────────────────────────────

@pytest.fixture(scope="session")
def error_report_dir() -> Path:
    """
    Directory where test cases should write their CSV error reports.
    Agent 3 scans this directory after pytest to collect all failures.
    """
    report_dir = _TEMP_DIR / "error_reports"
    report_dir.mkdir(exist_ok=True)
    # Clean old reports before each session
    for old in report_dir.glob("*.csv"):
        old.unlink(missing_ok=True)
    return report_dir