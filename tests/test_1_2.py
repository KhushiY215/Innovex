# tests/test_full_163_validation.py
"""
TC-1.2 — Full 163-Field Mandatory / Optional / Format Validation
─────────────────────────────────────────────────────────────────
Reads from  : staging_company_rows.csv  (built by conftest csv_path fixture)
Writes to   : error_report_dir/1_2_report.csv  (picked up by Agent 3)

Validation rules per field
──────────────────────────
  mandatory  → must not be null / empty string
  optional   → if present, must not be whitespace-only
  Special rules layered on top:
    incorporation_year  → must be purely numeric
    website_url         → must start with http:// or https://
    logo_url            → must start with http:// or https://
    primary_contact_email / contact_person_email → basic @ format
    glassdoor_rating / indeed_rating / google_rating → float 0-5
    annual_revenue / valuation → must be numeric (float/int)
    employee_size       → must be one of the allowed enum values
    ai_ml_adoption_level → must be one of the allowed enum values
    net_promoter_score  → numeric, -100 to 100
    market_share_percentage / churn_rate → numeric, 0 to 100
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# ─── output file name ─────────────────────────────────────────────────────────
OUTPUT_CSV = "1_2_report.csv"

# ─── enum allowed values ──────────────────────────────────────────────────────
EMPLOYEE_SIZE_VALUES = {
    "1-10", "11-50", "51-200", "201-500",
    "501-1000", "1001-5000", "5001-10000", "10000+",
}
AI_ML_VALUES = {"none", "basic", "intermediate", "advanced", "cutting_edge"}

# ─── field → (test_case_id, rule_type) mapping ────────────────────────────────
FIELD_TESTCASE_MAP: dict[str, tuple[str, str]] = {
    # ── mandatory ──────────────────────────────────────────────────────────
    "name":                      ("TC_VAL_001", "mandatory"),
    "logo_url":                  ("TC_VAL_003", "mandatory"),
    "category":                  ("TC_VAL_004", "mandatory"),
    "incorporation_year":        ("TC_VAL_005", "mandatory"),
    "overview_text":             ("TC_VAL_006", "mandatory"),
    "nature_of_company":         ("TC_VAL_007", "mandatory"),
    "headquarters_address":      ("TC_VAL_008", "mandatory"),
    "website_url":               ("TC_VAL_018", "mandatory"),
    "employee_size":             ("TC_VAL_023", "mandatory"),
    "ceo_name":                  ("TC_VAL_030", "mandatory"),

    # ── optional ───────────────────────────────────────────────────────────
    "short_name":                ("TC_VAL_002", "optional"),
    "operating_countries":       ("TC_VAL_009", "optional"),
    "office_count":              ("TC_VAL_010", "optional"),
    "primary_contact_email":     ("TC_VAL_020", "optional"),
    "primary_phone_number":      ("TC_VAL_021", "optional"),
    "board_members":             ("TC_VAL_032", "optional"),
    "glassdoor_rating":          ("TC_VAL_036", "optional"),
    "annual_revenue":            ("TC_VAL_043", "optional"),
    "valuation":                 ("TC_VAL_049", "optional"),
    "key_investors":             ("TC_VAL_050", "optional"),
    "focus_sectors":             ("TC_VAL_053", "optional"),
    "top_customers":             ("TC_VAL_055", "optional"),
    "offerings_description":     ("TC_VAL_056", "optional"),
    "sales_motion":              ("TC_VAL_057", "optional"),
    "core_value_proposition":    ("TC_VAL_058", "optional"),
    "key_competitors":           ("TC_VAL_059", "optional"),
    "market_share_percentage":   ("TC_VAL_060", "optional"),
    "technology_partners":       ("TC_VAL_061", "optional"),
    "tech_stack":                ("TC_VAL_062", "optional"),
    "intellectual_property":     ("TC_VAL_063", "optional"),
    "ai_ml_adoption_level":      ("TC_VAL_064", "optional"),
    "cybersecurity_posture":     ("TC_VAL_066", "optional"),
    "r_and_d_investment":        ("TC_VAL_067", "optional"),
    "regulatory_status":         ("TC_VAL_070", "optional"),
    "legal_issues":              ("TC_VAL_072", "optional"),
    "esg_ratings":               ("TC_VAL_076", "optional"),
    "sustainability_csr":        ("TC_VAL_077", "optional"),
    "carbon_footprint":          ("TC_VAL_079", "optional"),
    "ethical_sourcing":          ("TC_VAL_080", "optional"),
    "diversity_metrics":         ("TC_VAL_081", "optional"),
    "training_spend":            ("TC_VAL_083", "optional"),
    "avg_retention_tenure":      ("TC_VAL_086", "optional"),
    "esops_incentives":          ("TC_VAL_087", "optional"),
    "mentorship_availability":   ("TC_VAL_088", "optional"),
    "recent_news":               ("TC_VAL_089", "optional"),
    "brand_sentiment_score":     ("TC_VAL_090", "optional"),
    "event_participation":       ("TC_VAL_091", "optional"),
    "awards_recognitions":       ("TC_VAL_092", "optional"),
    "key_leaders":               ("TC_VAL_093", "optional"),
    "website_rating":            ("TC_VAL_094", "optional"),
    "recent_funding_rounds":     ("TC_VAL_095", "optional"),
    "total_capital_raised":      ("TC_VAL_096", "optional"),
    "partnership_ecosystem":     ("TC_VAL_098", "optional"),
    "core_values":               ("TC_VAL_102", "optional"),
    "work_culture_summary":      ("TC_VAL_103", "optional"),
    "feedback_culture":          ("TC_VAL_104", "optional"),
    "manager_quality":           ("TC_VAL_105", "optional"),
    "psychological_safety":      ("TC_VAL_106", "optional"),
    "customer_concentration_risk": ("TC_VAL_113", "optional"),
    "supply_chain_dependencies": ("TC_VAL_114", "optional"),
    "geopolitical_risks":        ("TC_VAL_115", "optional"),
    "office_locations":          ("TC_VAL_116", "optional"),
    "cab_policy":                ("TC_VAL_120", "optional"),
    "tools_access":              ("TC_VAL_121", "optional"),
    "flexibility_level":         ("TC_VAL_122", "optional"),
    "safety_policies":           ("TC_VAL_123", "optional"),
    "strategic_priorities":      ("TC_VAL_124", "optional"),
    "macro_risks":               ("TC_VAL_125", "optional"),
    "future_projections":        ("TC_VAL_126", "optional"),
    "product_pipeline":          ("TC_VAL_127", "optional"),
    "net_promoter_score":        ("TC_VAL_109", "optional"),
    "customer_lifetime_value":   ("TC_VAL_111", "optional"),
    "customer_acquisition_cost": ("TC_VAL_112", "optional"),
    "key_challenges_needs":      ("TC_VAL_125", "optional"),
    "competitive_advantages":    ("TC_VAL_059", "optional"),
    "vision_statement":          ("TC_VAL_128", "optional"),
    "mission_statement":         ("TC_VAL_161", "optional"),
    "mission_clarity":           ("TC_VAL_161", "optional"),
    "sustainability_csr":        ("TC_VAL_162", "optional"),
    "crisis_behavior":           ("TC_VAL_163", "optional"),
    "global_exposure":           ("TC_VAL_160", "optional"),
    "churn_rate":                ("TC_VAL_138", "optional"),
    "burn_rate":                 ("TC_VAL_139", "optional"),
    "runway_months":             ("TC_VAL_140", "optional"),
    "profitability_status":      ("TC_VAL_043", "optional"),
    "yoy_growth_rate":           ("TC_VAL_044", "optional"),
    "tam":                       ("TC_VAL_045", "optional"),
    "sam":                       ("TC_VAL_046", "optional"),
    "som":                       ("TC_VAL_047", "optional"),
    "linkedin_url":              ("TC_VAL_019", "optional"),
    "twitter_handle":            ("TC_VAL_019", "optional"),
    "facebook_url":              ("TC_VAL_019", "optional"),
    "instagram_url":             ("TC_VAL_019", "optional"),
    "ceo_linkedin_url":          ("TC_VAL_031", "optional"),
    "glassdoor_rating":          ("TC_VAL_036", "optional"),
    "indeed_rating":             ("TC_VAL_037", "optional"),
    "google_rating":             ("TC_VAL_038", "optional"),
    "contact_person_email":      ("TC_VAL_020", "optional"),
    "contact_person_phone":      ("TC_VAL_021", "optional"),
    "diversity_inclusion_score": ("TC_VAL_081", "optional"),
    "burnout_risk":              ("TC_VAL_086", "optional"),
    "layoff_history":            ("TC_VAL_095", "optional"),
    "exit_opportunities":        ("TC_VAL_097", "optional"),
    "skill_relevance":           ("TC_VAL_144", "optional"),
    "external_recognition":      ("TC_VAL_136", "optional"),
    "network_strength":          ("TC_VAL_136", "optional"),
    "company_maturity":          ("TC_VAL_104", "optional"),
    "brand_value":               ("TC_VAL_094", "optional"),
    "onboarding_quality":        ("TC_VAL_088", "optional"),
    "learning_culture":          ("TC_VAL_083", "optional"),
    "internal_mobility":         ("TC_VAL_135", "optional"),
    "promotion_clarity":         ("TC_VAL_085", "optional"),
    "role_clarity":              ("TC_VAL_106", "optional"),
    "early_ownership":           ("TC_VAL_087", "optional"),
    "typical_hours":             ("TC_VAL_086", "optional"),
    "leave_policy":              ("TC_VAL_082", "optional"),
    "family_health_insurance":   ("TC_VAL_082", "optional"),
    "relocation_support":        ("TC_VAL_082", "optional"),
    "lifestyle_benefits":        ("TC_VAL_082", "optional"),
    "bonus_predictability":      ("TC_VAL_085", "optional"),
    "fixed_vs_variable_pay":     ("TC_VAL_084", "optional"),
    "tech_adoption_rating":      ("TC_VAL_148", "optional"),
    "website_quality":           ("TC_VAL_018", "optional"),
    "website_traffic_rank":      ("TC_VAL_018", "optional"),
    "social_media_followers":    ("TC_VAL_019", "optional"),
    "pain_points_addressed":     ("TC_VAL_056", "optional"),
    "unique_differentiators":    ("TC_VAL_058", "optional"),
    "weaknesses_gaps":           ("TC_VAL_059", "optional"),
    "history_timeline":          ("TC_VAL_095", "optional"),
    "case_studies":              ("TC_VAL_078", "optional"),
    "go_to_market_strategy":     ("TC_VAL_057", "optional"),
    "innovation_roadmap":        ("TC_VAL_127", "optional"),
    "industry_associations":     ("TC_VAL_070", "optional"),
    "board_members":             ("TC_VAL_032", "optional"),
    "marketing_video_url":       ("TC_VAL_089", "optional"),
    "customer_testimonials":     ("TC_VAL_110", "optional"),
    "exit_strategy_history":     ("TC_VAL_097", "optional"),
    "benchmark_vs_peers":        ("TC_VAL_059", "optional"),
    "warm_intro_pathways":       ("TC_VAL_093", "optional"),
    "decision_maker_access":     ("TC_VAL_093", "optional"),
    "remote_policy_details":     ("TC_VAL_122", "optional"),
    "location_centrality":       ("TC_VAL_120", "optional"),
    "public_transport_access":   ("TC_VAL_120", "optional"),
    "airport_commute_time":      ("TC_VAL_120", "optional"),
    "office_zone_type":          ("TC_VAL_120", "optional"),
    "area_safety":               ("TC_VAL_123", "optional"),
    "infrastructure_safety":     ("TC_VAL_123", "optional"),
    "emergency_preparedness":    ("TC_VAL_123", "optional"),
    "health_support":            ("TC_VAL_082", "optional"),
    "exposure_quality":          ("TC_VAL_083", "optional"),
    "work_impact":               ("TC_VAL_087", "optional"),
    "execution_thinking_balance":("TC_VAL_104", "optional"),
    "automation_level":          ("TC_VAL_148", "optional"),
    "cross_functional_exposure": ("TC_VAL_135", "optional"),
    "client_quality":            ("TC_VAL_110", "optional"),
    "ethical_standards":         ("TC_VAL_080", "optional"),
    "overtime_expectations":     ("TC_VAL_086", "optional"),
    "weekend_work":              ("TC_VAL_086", "optional"),
    "burn_multiplier":           ("TC_VAL_139", "optional"),
    "cac_ltv_ratio":             ("TC_VAL_112", "optional"),
    "hiring_velocity":           ("TC_VAL_023", "optional"),
    "employee_turnover":         ("TC_VAL_023", "optional"),
}

# ─── special validation rules ─────────────────────────────────────────────────
URL_FIELDS    = {"website_url", "logo_url", "linkedin_url", "facebook_url",
                 "instagram_url", "ceo_linkedin_url", "marketing_video_url"}
EMAIL_FIELDS  = {"primary_contact_email", "contact_person_email"}
RATING_0_5    = {"glassdoor_rating", "indeed_rating", "google_rating"}
RATING_0_10   = {"website_rating", "tech_adoption_rating", "diversity_inclusion_score"}
NUMERIC_POS   = {"annual_revenue", "valuation", "total_capital_raised",
                 "r_and_d_investment", "training_spend", "carbon_footprint",
                 "burn_rate", "runway_months", "tam", "sam", "som",
                 "customer_acquisition_cost", "customer_lifetime_value",
                 "avg_retention_tenure", "typical_hours"}
RANGE_0_100   = {"market_share_percentage", "churn_rate"}
RANGE_NEG100  = {"net_promoter_score"}
RANGE_NEG1_1  = {"brand_sentiment_score"}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _validate_cell(column: str, value: str) -> str | None:
    """
    Returns an error message string if the value is invalid, else None.
    Applies rules in this order:
      1. Mandatory blank check
      2. Optional whitespace-only check
      3. Special field-level format / range rules
    """
    tc_id, rule = FIELD_TESTCASE_MAP.get(column, (None, "optional"))

    # ── 1. mandatory ──────────────────────────────────────────────────────
    if rule == "mandatory" and _is_blank(value):
        return "Mandatory Field Validation Error — value is empty or null"

    # ── 2. optional whitespace ────────────────────────────────────────────
    if rule == "optional" and value != "" and str(value).strip() == "":
        return "Whitespace-only value not allowed"

    # Skip further checks if blank (optional blank is OK)
    if _is_blank(value):
        return None

    val = str(value).strip()

    # ── 3a. incorporation_year ────────────────────────────────────────────
    if column == "incorporation_year":
        if not val.isdigit():
            return "Year must be numeric (e.g. 2001)"
        yr = int(val)
        if not (1800 <= yr <= 2025):
            return f"Year out of range 1800–2025 (got {yr})"

    # ── 3b. URL fields ────────────────────────────────────────────────────
    if column in URL_FIELDS:
        if not val.lower().startswith(("http://", "https://")):
            return "Invalid URL — must start with http:// or https://"

    # ── 3c. Email fields ──────────────────────────────────────────────────
    if column in EMAIL_FIELDS:
        if not EMAIL_RE.match(val):
            return "Invalid email format (expected user@domain.tld)"

    # ── 3d. Ratings 0–5 ──────────────────────────────────────────────────
    if column in RATING_0_5:
        try:
            fv = float(val)
            if not (0 <= fv <= 5):
                return f"Rating must be 0–5 (got {fv})"
        except ValueError:
            return f"Rating must be numeric (got {val!r})"

    # ── 3e. Ratings 0–10 ─────────────────────────────────────────────────
    if column in RATING_0_10:
        try:
            fv = float(val)
            if not (0 <= fv <= 10):
                return f"Rating must be 0–10 (got {fv})"
        except ValueError:
            return f"Rating must be numeric (got {val!r})"

    # ── 3f. Non-negative numerics ─────────────────────────────────────────
    if column in NUMERIC_POS:
        try:
            fv = float(val)
            if fv < 0:
                return f"Value must be >= 0 (got {fv})"
        except ValueError:
            return f"Must be numeric (got {val!r})"

    # ── 3g. Percentage 0–100 ─────────────────────────────────────────────
    if column in RANGE_0_100:
        try:
            fv = float(val)
            if not (0 <= fv <= 100):
                return f"Must be 0–100 (got {fv})"
        except ValueError:
            return f"Must be numeric 0–100 (got {val!r})"

    # ── 3h. NPS -100 to 100 ──────────────────────────────────────────────
    if column in RANGE_NEG100:
        try:
            fv = float(val)
            if not (-100 <= fv <= 100):
                return f"Must be -100 to 100 (got {fv})"
        except ValueError:
            return f"Must be numeric -100 to 100 (got {val!r})"

    # ── 3i. Sentiment -1 to 1 ────────────────────────────────────────────
    if column in RANGE_NEG1_1:
        try:
            fv = float(val)
            if not (-1.0 <= fv <= 1.0):
                return f"Must be -1.0 to 1.0 (got {fv})"
        except ValueError:
            return f"Must be numeric -1.0 to 1.0 (got {val!r})"

    # ── 3j. employee_size enum ────────────────────────────────────────────
    if column == "employee_size":
        if val not in EMPLOYEE_SIZE_VALUES:
            return (
                f"Must be one of {sorted(EMPLOYEE_SIZE_VALUES)} (got {val!r})"
            )

    # ── 3k. ai_ml_adoption_level enum ────────────────────────────────────
    if column == "ai_ml_adoption_level":
        if val not in AI_ML_VALUES:
            return f"Must be one of {sorted(AI_ML_VALUES)} (got {val!r})"

    return None   # all checks passed


# ─── fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def load_csv(csv_path):
    """Load staging_company_rows.csv into a DataFrame."""
    assert csv_path.exists(), f"CSV not found at {csv_path}"
    df = pd.read_csv(csv_path, dtype=str)
    df.fillna("", inplace=True)
    return df


# ─── main test ────────────────────────────────────────────────────────────────

def test_full_163_validation(load_csv, error_report_dir):
    """
    TC-1.2 — Full 163-Field Validation
    Validates all mapped fields in staging_company_rows.csv.
    Writes 1_2_report.csv to error_report_dir on failure.
    """
    df = load_csv
    failed_results = []

    for index, row in df.iterrows():
        row_number = index + 1

        for column in df.columns:
            if column not in FIELD_TESTCASE_MAP:
                continue

            tc_id, _ = FIELD_TESTCASE_MAP[column]
            value     = row[column]
            message   = _validate_cell(column, value)

            if message:
                failed_results.append({
                    "row_number":    row_number,
                    "column_name":   column,
                    "test_case_id":  tc_id,
                    "input_value":   value,
                    "error_message": message,
                })

    # ── write report ──────────────────────────────────────────────────────
    failed_df   = pd.DataFrame(failed_results)
    report_path = error_report_dir / OUTPUT_CSV
    failed_df.to_csv(report_path, index=False)

    total_failures = len(failed_df)
    print(f"\nTotal TC-1.2 validation failures : {total_failures}")
    print(f"Failure report saved to          : {report_path}")

    assert total_failures == 0, (
        f"{total_failures} validation failure(s) found across 163 fields. "
        f"See {report_path}"
    )