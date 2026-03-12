# tests/test_company_name.py
"""
TC-1.1 — Company Name Validation
──────────────────────────────────
Test cases for the `name` field in staging_company.

Reads from staging_company_rows.csv (written by conftest csv_path fixture).
Writes a CSV error report to error_report_dir on failure.
Failure details are picked up by Agent 3 and merged into the feedback loop.

Test IDs
────────
  TC-1.1-COMPANYNAME-01  Validate full legal company name
  TC-1.1-COMPANYNAME-02  Validate legally formatted name with punctuation
  TC-1.1-COMPANYNAME-03  Validate multinational corporate naming
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

# ─── regex ────────────────────────────────────────────────────────────────────
# Allows: word chars, spaces, & . , - ( ) standard apostrophe ' smart apostrophe '
# Latin extended (accented chars), trailing spaces allowed.

COMPANY_NAME_REGEX = re.compile(
    r"^[\w\s&.,\-\(\)''\u00C0-\u017F]+$"
)

# ─── test case registry ───────────────────────────────────────────────────────

TEST_CASES = {
    "TC-1.1-COMPANYNAME-01": {
        "description": "Validate full legal company name",
        "example":     "Microsoft Corporation",
    },
    "TC-1.1-COMPANYNAME-02": {
        "description": "Validate legally formatted name with punctuation",
        "example":     "Tesla, Inc.",
    },
    "TC-1.1-COMPANYNAME-03": {
        "description": "Validate multinational corporate naming",
        "example":     "Apple Inc.",
    },
}


# ─── validation logic ─────────────────────────────────────────────────────────

def validate_company_name(value: str) -> str | None:
    """
    Returns an error string if `value` is invalid, else None.

    Rules
    ─────
    1. Must not be empty.
    2. Must only contain allowed characters (COMPANY_NAME_REGEX).
    3. Must be at least 2 characters long.
    4. Must not be ALL digits.
    5. Must not exceed 255 characters.
    """
    if not value or not value.strip():
        return "Company name cannot be empty"

    if len(value.strip()) < 2:
        return "Company name too short (minimum 2 characters)"

    if len(value) > 255:
        return f"Company name too long ({len(value)} chars, max 255)"

    if value.strip().isdigit():
        return "Company name must not be purely numeric"

    if not COMPANY_NAME_REGEX.match(value.strip()):
        bad = [c for c in value if not re.match(r"[\w\s&.,\-\(\)''\u00C0-\u017F]", c)]
        return f"Invalid characters detected: {bad}"

    return None


# ─── tests ────────────────────────────────────────────────────────────────────

def test_company_name_validation(csv_path: Path, error_report_dir: Path):
    """
    TC-1.1-COMPANYNAME-01/02/03
    Validates the `name` column in staging_company_rows.csv against all
    three company-name test cases.
    Writes error report to error_report_dir/1_1_report.csv on failure.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    if "name" not in df.columns:
        pytest.fail("Column 'name' not found in CSV — check conftest csv_path fixture")

    errors = []

    for row_index, row in df.iterrows():
        company_name = row["name"]
        error = validate_company_name(company_name)

        if error:
            # Log one row for each TC so all three IDs appear in the report
            for tc_id, tc_meta in TEST_CASES.items():
                errors.append({
                    "Row":         row_index + 1,
                    "Test ID":     tc_id,
                    "Description": tc_meta["description"],
                    "Example":     tc_meta["example"],
                    "Field":       "name",
                    "Value":       company_name,
                    "Error":       error,
                })
            break  # one company per run — only need one failing entry

    if errors:
        report_path = error_report_dir / "1_1_report.csv"
        pd.DataFrame(errors).to_csv(report_path, index=False)
        pytest.fail(
            f"{len(errors)} company name validation error(s) found. "
            f"Report → {report_path}\n"
            f"Failing value: {errors[0]['Value']!r}\n"
            f"Reason: {errors[0]['Error']}"
        )


def test_company_name_not_empty(company: dict):
    """
    TC-1.1-COMPANYNAME-01 (direct JSON check)
    name field must be present and non-empty in the company dict.
    """
    value = company.get("name", "")
    assert value and value.strip(), \
        "TC-1.1-COMPANYNAME-01: 'name' must be a non-empty string. Got: {!r}".format(value)


def test_company_name_no_invalid_chars(company: dict):
    """
    TC-1.1-COMPANYNAME-02
    name must only contain legal corporate name characters.
    """
    value = company.get("name", "")
    if not value:
        pytest.skip("name is empty — covered by test_company_name_not_empty")

    assert COMPANY_NAME_REGEX.match(value.strip()), (
        f"TC-1.1-COMPANYNAME-02: 'name' contains invalid characters.\n"
        f"  Value: {value!r}\n"
        f"  Allowed: word chars, spaces, & . , - ( ) apostrophes, accented letters\n"
        f"  Example of valid name: 'Apple Inc.'"
    )


def test_company_name_length(company: dict):
    """
    TC-1.1-COMPANYNAME-03
    name length must be between 2 and 255 characters.
    """
    value = company.get("name", "")
    if not value:
        pytest.skip("name is empty — covered by test_company_name_not_empty")

    assert 2 <= len(value.strip()) <= 255, (
        f"TC-1.1-COMPANYNAME-03: 'name' length out of range.\n"
        f"  Value: {value!r}\n"
        f"  Length: {len(value.strip())} (must be 2–255)"
    )


def test_company_name_not_numeric(company: dict):
    """
    TC-1.1-COMPANYNAME-03 (extra guard)
    name must not be purely numeric.
    """
    value = company.get("name", "").strip()
    if not value:
        pytest.skip("name is empty")

    assert not value.isdigit(), (
        f"'name' must not be purely numeric. Got: {value!r}"
    )