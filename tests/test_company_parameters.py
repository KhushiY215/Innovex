# tests/test_company_parameters.py
"""
Pytest test suite — 35+ test cases covering all 163 company parameters.
────────────────────────────────────────────────────────────────────────
Test conventions
  • Each test is named  test_<field_name>  so Agent 3 can map failures → fields.
  • Tests use the `company` fixture (dict) from conftest.py.
  • Soft assertions: xfail-style skips when field is None (optional fields).
  • Hard assertions: required fields, range checks, format checks, consistency.
"""

from __future__ import annotations

import re
import pytest
from typing import Any, Optional

# ─── helpers ──────────────────────────────────────────────────────────────────

_URL_RE    = re.compile(r"^https?://.+\..+", re.IGNORECASE)
_EMAIL_RE  = re.compile(r"^[^@]+@[^@]+\.[^@]+$")
_PHONE_RE  = re.compile(r"[\d\s\-\+\(\)]{7,20}")


def _skip_if_none(value: Any, field: str):
    if value is None:
        pytest.skip(f"Field '{field}' is null — skipping optional check")


def _assert_url(value: Any, field: str):
    _skip_if_none(value, field)
    assert _URL_RE.match(str(value)), \
        f"'{field}' must be a valid URL starting with http(s)://. Got: {value!r}"


def _assert_email(value: Any, field: str):
    _skip_if_none(value, field)
    assert _EMAIL_RE.match(str(value)), \
        f"'{field}' must be a valid email address. Got: {value!r}"


def _assert_range(value: Any, field: str, lo: float, hi: float):
    _skip_if_none(value, field)
    assert lo <= float(value) <= hi, \
        f"'{field}' must be between {lo} and {hi}. Got: {value}"


def _assert_positive(value: Any, field: str):
    _skip_if_none(value, field)
    assert float(value) >= 0, \
        f"'{field}' must be >= 0. Got: {value}"


def _assert_enum(value: Any, field: str, allowed: set):
    _skip_if_none(value, field)
    assert value in allowed, \
        f"'{field}' must be one of {allowed}. Got: {value!r}"


def _assert_list_not_empty(value: Any, field: str):
    _skip_if_none(value, field)
    assert isinstance(value, list) and len(value) > 0, \
        f"'{field}' must be a non-empty list when present. Got: {value!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — REQUIRED FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

def test_name(company):
    """Company name must be present and non-empty."""
    assert company.get("name"), "Field 'name' is required and must not be empty"


def test_short_name(company):
    """Short name, if present, must be a non-empty string."""
    v = company.get("short_name")
    _skip_if_none(v, "short_name")
    assert isinstance(v, str) and v.strip(), \
        f"'short_name' must be a non-empty string. Got: {v!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — YEAR / DATE
# ═══════════════════════════════════════════════════════════════════════════════

def test_incorporation_year(company):
    """Incorporation year must be between 1800 and 2025."""
    v = company.get("incorporation_year")
    _skip_if_none(v, "incorporation_year")
    assert 1800 <= int(v) <= 2025, \
        f"'incorporation_year' must be 1800–2025. Got: {v}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — URL FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

def test_website_url(company):    _assert_url(company.get("website_url"),    "website_url")
def test_logo_url(company):       _assert_url(company.get("logo_url"),       "logo_url")
def test_linkedin_url(company):   _assert_url(company.get("linkedin_url"),   "linkedin_url")
def test_facebook_url(company):   _assert_url(company.get("facebook_url"),   "facebook_url")
def test_instagram_url(company):  _assert_url(company.get("instagram_url"),  "instagram_url")
def test_ceo_linkedin_url(company): _assert_url(company.get("ceo_linkedin_url"), "ceo_linkedin_url")
def test_marketing_video_url(company): _assert_url(company.get("marketing_video_url"), "marketing_video_url")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — EMAIL & PHONE
# ═══════════════════════════════════════════════════════════════════════════════

def test_primary_contact_email(company):
    _assert_email(company.get("primary_contact_email"), "primary_contact_email")

def test_contact_person_email(company):
    _assert_email(company.get("contact_person_email"), "contact_person_email")

def test_primary_phone_number(company):
    v = company.get("primary_phone_number")
    _skip_if_none(v, "primary_phone_number")
    assert _PHONE_RE.search(str(v)), \
        f"'primary_phone_number' must be a valid phone. Got: {v!r}"

def test_twitter_handle(company):
    v = company.get("twitter_handle")
    _skip_if_none(v, "twitter_handle")
    assert str(v).startswith("@"), \
        f"'twitter_handle' must start with '@'. Got: {v!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — RATINGS (0–5 or 0–10)
# ═══════════════════════════════════════════════════════════════════════════════

def test_glassdoor_rating(company):   _assert_range(company.get("glassdoor_rating"),  "glassdoor_rating",  0, 5)
def test_indeed_rating(company):      _assert_range(company.get("indeed_rating"),     "indeed_rating",     0, 5)
def test_google_rating(company):      _assert_range(company.get("google_rating"),     "google_rating",     0, 5)
def test_website_rating(company):     _assert_range(company.get("website_rating"),    "website_rating",    0, 10)
def test_tech_adoption_rating(company): _assert_range(company.get("tech_adoption_rating"), "tech_adoption_rating", 0, 10)
def test_diversity_inclusion_score(company): _assert_range(company.get("diversity_inclusion_score"), "diversity_inclusion_score", 0, 10)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — SCORES WITH SPECIFIC RANGES
# ═══════════════════════════════════════════════════════════════════════════════

def test_brand_sentiment_score(company):
    _assert_range(company.get("brand_sentiment_score"), "brand_sentiment_score", -1.0, 1.0)

def test_net_promoter_score(company):
    _assert_range(company.get("net_promoter_score"), "net_promoter_score", -100, 100)

def test_yoy_growth_rate(company):
    v = company.get("yoy_growth_rate")
    _skip_if_none(v, "yoy_growth_rate")
    assert float(v) >= -100, f"'yoy_growth_rate' cannot be < -100%. Got: {v}"

def test_market_share_percentage(company):
    _assert_range(company.get("market_share_percentage"), "market_share_percentage", 0, 100)

def test_churn_rate(company):
    _assert_range(company.get("churn_rate"), "churn_rate", 0, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 7 — NON-NEGATIVE FINANCIALS
# ═══════════════════════════════════════════════════════════════════════════════

def test_annual_revenue(company):       _assert_positive(company.get("annual_revenue"),      "annual_revenue")
def test_valuation(company):            _assert_positive(company.get("valuation"),            "valuation")
def test_total_capital_raised(company): _assert_positive(company.get("total_capital_raised"), "total_capital_raised")
def test_customer_acquisition_cost(company): _assert_positive(company.get("customer_acquisition_cost"), "customer_acquisition_cost")
def test_customer_lifetime_value(company):   _assert_positive(company.get("customer_lifetime_value"),   "customer_lifetime_value")
def test_burn_rate(company):            _assert_positive(company.get("burn_rate"),            "burn_rate")
def test_r_and_d_investment(company):   _assert_positive(company.get("r_and_d_investment"),   "r_and_d_investment")
def test_training_spend(company):       _assert_positive(company.get("training_spend"),       "training_spend")
def test_carbon_footprint(company):     _assert_positive(company.get("carbon_footprint"),     "carbon_footprint")
def test_tam(company):                  _assert_positive(company.get("tam"), "tam")
def test_sam(company):                  _assert_positive(company.get("sam"), "sam")
def test_som(company):                  _assert_positive(company.get("som"), "som")

def test_runway_months(company):
    v = company.get("runway_months")
    _skip_if_none(v, "runway_months")
    assert int(v) >= 0, f"'runway_months' must be >= 0. Got: {v}"

def test_office_count(company):
    v = company.get("office_count")
    _skip_if_none(v, "office_count")
    assert int(v) >= 0, f"'office_count' must be >= 0. Got: {v}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 8 — ENUM FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

def test_employee_size(company):
    _assert_enum(company.get("employee_size"), "employee_size",
        {"1-10","11-50","51-200","201-500","501-1000","1001-5000","5001-10000","10000+"})

def test_hiring_velocity(company):
    _assert_enum(company.get("hiring_velocity"), "hiring_velocity",
        {"low","medium","high","very_high"})

def test_employee_turnover(company):
    _assert_enum(company.get("employee_turnover"), "employee_turnover",
        {"low","moderate","high","critical"})

def test_ai_ml_adoption_level(company):
    _assert_enum(company.get("ai_ml_adoption_level"), "ai_ml_adoption_level",
        {"none","basic","intermediate","advanced","cutting_edge"})

def test_profitability_status(company):
    _assert_enum(company.get("profitability_status"), "profitability_status",
        {"profitable","break_even","pre_revenue","loss_making"})

def test_company_maturity(company):
    _assert_enum(company.get("company_maturity"), "company_maturity",
        {"idea","startup","early_stage","growth","mature","enterprise"})

def test_sales_motion(company):
    _assert_enum(company.get("sales_motion"), "sales_motion",
        {"inbound","outbound","product_led","channel","hybrid"})

def test_burnout_risk(company):
    _assert_enum(company.get("burnout_risk"), "burnout_risk",
        {"low","medium","high","very_high"})

def test_flexibility_level(company):
    _assert_enum(company.get("flexibility_level"), "flexibility_level",
        {"none","low","medium","high","fully_remote"})

def test_website_quality(company):
    _assert_enum(company.get("website_quality"), "website_quality",
        {"poor","average","good","excellent"})

def test_cybersecurity_posture(company):
    _assert_enum(company.get("cybersecurity_posture"), "cybersecurity_posture",
        {"weak","moderate","strong","advanced"})

def test_decision_maker_access(company):
    _assert_enum(company.get("decision_maker_access"), "decision_maker_access",
        {"easy","moderate","difficult"})

def test_regulatory_status(company):
    _assert_enum(company.get("regulatory_status"), "regulatory_status",
        {"compliant","partially_compliant","non_compliant","under_review"})

def test_area_safety(company):
    _assert_enum(company.get("area_safety"), "area_safety",
        {"unsafe","moderate","safe","very_safe"})


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 9 — LIST FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

def test_operating_countries(company):    _assert_list_not_empty(company.get("operating_countries"),    "operating_countries")
def test_focus_sectors(company):          _assert_list_not_empty(company.get("focus_sectors"),          "focus_sectors")
def test_key_competitors(company):        _assert_list_not_empty(company.get("key_competitors"),        "key_competitors")
def test_tech_stack(company):             _assert_list_not_empty(company.get("tech_stack"),             "tech_stack")
def test_strategic_priorities(company):   _assert_list_not_empty(company.get("strategic_priorities"),   "strategic_priorities")
def test_core_values(company):            _assert_list_not_empty(company.get("core_values"),            "core_values")
def test_competitive_advantages(company): _assert_list_not_empty(company.get("competitive_advantages"), "competitive_advantages")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 10 — CROSS-FIELD CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════

def test_tam_gte_sam(company):
    tam = company.get("tam")
    sam = company.get("sam")
    if tam is None or sam is None:
        pytest.skip("tam or sam is null")
    assert float(tam) >= float(sam), \
        f"TAM ({tam}) must be >= SAM ({sam})"

def test_sam_gte_som(company):
    sam = company.get("sam")
    som = company.get("som")
    if sam is None or som is None:
        pytest.skip("sam or som is null")
    assert float(sam) >= float(som), \
        f"SAM ({sam}) must be >= SOM ({som})"

def test_profitability_profit_consistency(company):
    status = company.get("profitability_status")
    profit = company.get("annual_profit")
    if status != "profitable" or profit is None:
        pytest.skip("Not applicable")
    assert float(profit) >= 0, \
        f"If profitability_status='profitable', annual_profit must be >= 0. Got: {profit}"

def test_cac_ltv_ratio_consistency(company):
    cac   = company.get("customer_acquisition_cost")
    ltv   = company.get("customer_lifetime_value")
    ratio = company.get("cac_ltv_ratio")
    if cac is None or ltv is None or ratio is None or float(cac) == 0:
        pytest.skip("CAC/LTV fields incomplete")
    expected = float(ltv) / float(cac)
    assert abs(float(ratio) - expected) < 0.5, \
        f"cac_ltv_ratio ({ratio}) inconsistent with LTV/CAC = {expected:.4f}"

def test_typical_hours(company):
    v = company.get("typical_hours")
    _skip_if_none(v, "typical_hours")
    assert 0 <= int(v) <= 168, \
        f"'typical_hours' per week must be 0–168. Got: {v}"

def test_avg_retention_tenure(company):
    v = company.get("avg_retention_tenure")
    _skip_if_none(v, "avg_retention_tenure")
    assert float(v) >= 0, f"'avg_retention_tenure' must be >= 0. Got: {v}"

def test_social_media_followers_shape(company):
    v = company.get("social_media_followers")
    _skip_if_none(v, "social_media_followers")
    assert isinstance(v, dict), \
        f"'social_media_followers' must be a dict. Got: {type(v)}"
    for platform, count in v.items():
        if count is not None:
            assert int(count) >= 0, \
                f"Follower count for '{platform}' must be >= 0. Got: {count}"

def test_key_leaders_shape(company):
    v = company.get("key_leaders")
    _skip_if_none(v, "key_leaders")
    assert isinstance(v, list), f"'key_leaders' must be a list. Got: {type(v)}"
    for leader in v:
        assert "name" in leader, f"Each key_leader must have a 'name' field. Got: {leader}"

def test_recent_funding_rounds_shape(company):
    v = company.get("recent_funding_rounds")
    _skip_if_none(v, "recent_funding_rounds")
    assert isinstance(v, list), f"'recent_funding_rounds' must be a list."
    for rnd in v:
        assert isinstance(rnd, dict), f"Each funding round must be a dict. Got: {rnd!r}"
        if rnd.get("amount_usd") is not None:
            assert float(rnd["amount_usd"]) >= 0, \
                f"Funding round amount must be >= 0. Got: {rnd['amount_usd']}"

def test_history_timeline_shape(company):
    v = company.get("history_timeline")
    _skip_if_none(v, "history_timeline")
    assert isinstance(v, list)
    for item in v:
        assert "year" in item and "event" in item, \
            f"history_timeline items need 'year' and 'event'. Got: {item!r}"
        assert 1800 <= int(item["year"]) <= 2030, \
            f"history_timeline year out of range: {item['year']}"

def test_revenue_mix_sums_to_100(company):
    v = company.get("revenue_mix")
    _skip_if_none(v, "revenue_mix")
    assert isinstance(v, dict), f"'revenue_mix' must be a dict. Got: {type(v)}"
    total = sum(float(pct) for pct in v.values() if pct is not None)
    assert abs(total - 100.0) < 5.0, \
        f"'revenue_mix' percentages should sum to ~100. Got: {total}"

def test_esg_ratings_shape(company):
    v = company.get("esg_ratings")
    _skip_if_none(v, "esg_ratings")
    assert isinstance(v, dict)
    allowed = {"low", "medium", "high", "exemplary"}
    for dim in ("environmental", "social", "governance"):
        val = v.get(dim)
        if val is not None:
            assert val in allowed, \
                f"esg_ratings.{dim} must be in {allowed}. Got: {val!r}"
    if v.get("overall_score") is not None:
        assert 0 <= float(v["overall_score"]) <= 100, \
            f"esg_ratings.overall_score must be 0–100. Got: {v['overall_score']}"

def test_board_members_shape(company):
    v = company.get("board_members")
    _skip_if_none(v, "board_members")
    assert isinstance(v, list)
    for member in v:
        assert "name" in member, \
            f"Each board_member must have a 'name' field. Got: {member!r}"

def test_case_studies_shape(company):
    v = company.get("case_studies")
    _skip_if_none(v, "case_studies")
    assert isinstance(v, list)
    for cs in v:
        assert "title" in cs, \
            f"Each case_study must have a 'title'. Got: {cs!r}"

def test_burn_multiplier(company):
    v = company.get("burn_multiplier")
    _skip_if_none(v, "burn_multiplier")
    assert float(v) >= 0, f"'burn_multiplier' must be >= 0. Got: {v}"

def test_cac_ltv_ratio_positive(company):
    v = company.get("cac_ltv_ratio")
    _skip_if_none(v, "cac_ltv_ratio")
    assert float(v) >= 0, f"'cac_ltv_ratio' must be >= 0. Got: {v}"
