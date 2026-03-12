# schema/company_schema.py
"""
Pydantic v2 schema covering all 163 staging_company parameters.
Used for:
  • Agent 1  — per-LLM output validation
  • Agent 2  — consolidated output validation
  • Agent 3  — pre-test structural check
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class EmployeeSizeEnum(str, Enum):
    micro        = "1-10"
    small        = "11-50"
    small_med    = "51-200"
    medium       = "201-500"
    med_large    = "501-1000"
    large        = "1001-5000"
    very_large   = "5001-10000"
    enterprise   = "10000+"


class HiringVelocityEnum(str, Enum):
    low       = "low"
    medium    = "medium"
    high      = "high"
    very_high = "very_high"


class EmployeeTurnoverEnum(str, Enum):
    low      = "low"
    moderate = "moderate"
    high     = "high"
    critical = "critical"


class AIMLAdoptionEnum(str, Enum):
    none         = "none"
    basic        = "basic"
    intermediate = "intermediate"
    advanced     = "advanced"
    cutting_edge = "cutting_edge"


class ProfitabilityEnum(str, Enum):
    profitable   = "profitable"
    break_even   = "break_even"
    pre_revenue  = "pre_revenue"
    loss_making  = "loss_making"


class CompanyMaturityEnum(str, Enum):
    idea       = "idea"
    startup    = "startup"
    early      = "early_stage"
    growth     = "growth"
    mature     = "mature"
    enterprise = "enterprise"


class SalesMotionEnum(str, Enum):
    inbound      = "inbound"
    outbound     = "outbound"
    product_led  = "product_led"
    channel      = "channel"
    hybrid       = "hybrid"


class BurnoutRiskEnum(str, Enum):
    low       = "low"
    medium    = "medium"
    high      = "high"
    very_high = "very_high"


class FlexibilityEnum(str, Enum):
    none          = "none"
    low           = "low"
    medium        = "medium"
    high          = "high"
    fully_remote  = "fully_remote"


class WebsiteQualityEnum(str, Enum):
    poor      = "poor"
    average   = "average"
    good      = "good"
    excellent = "excellent"


class CybersecurityPostureEnum(str, Enum):
    weak     = "weak"
    moderate = "moderate"
    strong   = "strong"
    advanced = "advanced"


class ESGRatingEnum(str, Enum):
    low       = "low"
    medium    = "medium"
    high      = "high"
    exemplary = "exemplary"


class DecisionMakerAccessEnum(str, Enum):
    easy      = "easy"
    moderate  = "moderate"
    difficult = "difficult"


class RegulatoryStatusEnum(str, Enum):
    compliant     = "compliant"
    partially     = "partially_compliant"
    non_compliant = "non_compliant"
    under_review  = "under_review"


class AreaSafetyEnum(str, Enum):
    unsafe   = "unsafe"
    moderate = "moderate"
    safe     = "safe"
    very_safe = "very_safe"


# ═══════════════════════════════════════════════════════════════════════════════
# NESTED MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class KeyLeader(BaseModel):
    name:       Optional[str] = None
    title:      Optional[str] = None
    linkedin:   Optional[str] = None
    email:      Optional[str] = None


class FundingRound(BaseModel):
    round_name:  Optional[str] = None
    amount_usd:  Optional[float] = None
    date:        Optional[str] = None
    investors:   Optional[List[str]] = None


class HistoryEvent(BaseModel):
    year:        Optional[int] = None
    event:       Optional[str] = None


class SocialMediaFollowers(BaseModel):
    linkedin:  Optional[int] = None
    twitter:   Optional[int] = None
    facebook:  Optional[int] = None
    instagram: Optional[int] = None
    youtube:   Optional[int] = None


class ESGRatings(BaseModel):
    environmental: Optional[ESGRatingEnum] = None
    social:        Optional[ESGRatingEnum] = None
    governance:    Optional[ESGRatingEnum] = None
    overall_score: Optional[float] = None


class DiversityMetrics(BaseModel):
    gender_ratio:    Optional[str] = None
    minority_pct:    Optional[float] = None
    leadership_diversity: Optional[str] = None


class BoardMember(BaseModel):
    name:       Optional[str] = None
    title:      Optional[str] = None
    linkedin:   Optional[str] = None


class CaseStudy(BaseModel):
    title:        Optional[str] = None
    client:       Optional[str] = None
    outcome:      Optional[str] = None
    url:          Optional[str] = None


class RecentNews(BaseModel):
    headline:     Optional[str] = None
    date:         Optional[str] = None
    source:       Optional[str] = None
    url:          Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCHEMA — 163 PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

class CompanySchema(BaseModel):
    """
    Full schema for staging_company table — 163 parameters.
    name is the only strictly required field; all others are Optional
    but carry validators for range / format / enum correctness.
    """

    # ── 1. Identity ───────────────────────────────────────────────────────
    name:                   str = Field(..., min_length=1, description="Official company name")
    short_name:             Optional[str] = None
    logo_url:               Optional[str] = None
    category:               Optional[str] = None
    incorporation_year:     Optional[int] = Field(None, ge=1800, le=2025)
    overview_text:          Optional[str] = None
    nature_of_company:      Optional[str] = None

    # ── 2. Presence ───────────────────────────────────────────────────────
    headquarters_address:   Optional[str] = None
    operating_countries:    Optional[List[str]] = None
    office_count:           Optional[int] = Field(None, ge=0)
    office_locations:       Optional[List[str]] = None

    # ── 3. Workforce ──────────────────────────────────────────────────────
    employee_size:          Optional[EmployeeSizeEnum] = None
    hiring_velocity:        Optional[HiringVelocityEnum] = None
    employee_turnover:      Optional[EmployeeTurnoverEnum] = None
    avg_retention_tenure:   Optional[float] = Field(None, ge=0, description="Average years")

    # ── 4. Business Context ───────────────────────────────────────────────
    pain_points_addressed:  Optional[List[str]] = None
    focus_sectors:          Optional[List[str]] = None
    offerings_description:  Optional[str] = None
    top_customers:          Optional[List[str]] = None
    core_value_proposition: Optional[str] = None
    vision_statement:       Optional[str] = None
    mission_statement:      Optional[str] = None
    core_values:            Optional[List[str]] = None
    unique_differentiators: Optional[List[str]] = None
    competitive_advantages: Optional[List[str]] = None
    weaknesses_gaps:        Optional[List[str]] = None
    key_challenges_needs:   Optional[List[str]] = None
    key_competitors:        Optional[List[str]] = None
    technology_partners:    Optional[List[str]] = None

    # ── 5. History & News ─────────────────────────────────────────────────
    history_timeline:       Optional[List[HistoryEvent]] = None
    recent_news:            Optional[List[RecentNews]] = None

    # ── 6. Digital Presence ───────────────────────────────────────────────
    website_url:            Optional[str] = None
    website_quality:        Optional[WebsiteQualityEnum] = None
    website_rating:         Optional[float] = Field(None, ge=0, le=10)
    website_traffic_rank:   Optional[int] = Field(None, ge=0)
    social_media_followers: Optional[SocialMediaFollowers] = None
    glassdoor_rating:       Optional[float] = Field(None, ge=0, le=5)
    indeed_rating:          Optional[float] = Field(None, ge=0, le=5)
    google_rating:          Optional[float] = Field(None, ge=0, le=5)
    linkedin_url:           Optional[str] = None
    twitter_handle:         Optional[str] = None
    facebook_url:           Optional[str] = None
    instagram_url:          Optional[str] = None

    # ── 7. Leadership & Contacts ──────────────────────────────────────────
    ceo_name:               Optional[str] = None
    ceo_linkedin_url:       Optional[str] = None
    key_leaders:            Optional[List[KeyLeader]] = None
    warm_intro_pathways:    Optional[List[str]] = None
    decision_maker_access:  Optional[DecisionMakerAccessEnum] = None
    primary_contact_email:  Optional[str] = None
    primary_phone_number:   Optional[str] = None
    contact_person_name:    Optional[str] = None
    contact_person_title:   Optional[str] = None
    contact_person_email:   Optional[str] = None
    contact_person_phone:   Optional[str] = None

    # ── 8. Brand & Recognition ────────────────────────────────────────────
    awards_recognitions:    Optional[List[str]] = None
    brand_sentiment_score:  Optional[float] = Field(None, ge=-1.0, le=1.0)
    event_participation:    Optional[List[str]] = None

    # ── 9. Legal & Regulatory ─────────────────────────────────────────────
    regulatory_status:      Optional[RegulatoryStatusEnum] = None
    legal_issues:           Optional[List[str]] = None

    # ── 10. Financials ────────────────────────────────────────────────────
    annual_revenue:         Optional[float] = Field(None, ge=0)
    annual_profit:          Optional[float] = None
    revenue_mix:            Optional[Dict[str, float]] = None
    valuation:              Optional[float] = Field(None, ge=0)
    yoy_growth_rate:        Optional[float] = Field(None, ge=-100)
    profitability_status:   Optional[ProfitabilityEnum] = None
    market_share_percentage:Optional[float] = Field(None, ge=0, le=100)
    key_investors:          Optional[List[str]] = None
    recent_funding_rounds:  Optional[List[FundingRound]] = None
    total_capital_raised:   Optional[float] = Field(None, ge=0)
    esg_ratings:            Optional[ESGRatings] = None

    # ── 11. GTM & Sales ───────────────────────────────────────────────────
    sales_motion:               Optional[SalesMotionEnum] = None
    customer_acquisition_cost:  Optional[float] = Field(None, ge=0)
    customer_lifetime_value:    Optional[float] = Field(None, ge=0)
    cac_ltv_ratio:              Optional[float] = Field(None, ge=0)
    churn_rate:                 Optional[float] = Field(None, ge=0, le=100)
    net_promoter_score:         Optional[float] = Field(None, ge=-100, le=100)
    customer_concentration_risk:Optional[str] = None

    # ── 12. Burn & Runway ─────────────────────────────────────────────────
    burn_rate:              Optional[float] = Field(None, ge=0)
    runway_months:          Optional[int] = Field(None, ge=0)
    burn_multiplier:        Optional[float] = Field(None, ge=0)

    # ── 13. Technology & IP ───────────────────────────────────────────────
    intellectual_property:  Optional[List[str]] = None
    r_and_d_investment:     Optional[float] = Field(None, ge=0)
    ai_ml_adoption_level:   Optional[AIMLAdoptionEnum] = None
    tech_stack:             Optional[List[str]] = None
    cybersecurity_posture:  Optional[CybersecurityPostureEnum] = None

    # ── 14. Risk ──────────────────────────────────────────────────────────
    supply_chain_dependencies: Optional[List[str]] = None
    geopolitical_risks:     Optional[List[str]] = None
    macro_risks:            Optional[List[str]] = None

    # ── 15. People & Culture ──────────────────────────────────────────────
    diversity_metrics:      Optional[DiversityMetrics] = None
    remote_policy_details:  Optional[str] = None
    training_spend:         Optional[float] = Field(None, ge=0)

    # ── 16. Ecosystem ─────────────────────────────────────────────────────
    partnership_ecosystem:  Optional[List[str]] = None
    exit_strategy_history:  Optional[List[str]] = None

    # ── 17. ESG / Ethics ──────────────────────────────────────────────────
    carbon_footprint:       Optional[float] = Field(None, ge=0)
    ethical_sourcing:       Optional[str] = None

    # ── 18. Strategic ─────────────────────────────────────────────────────
    benchmark_vs_peers:     Optional[str] = None
    future_projections:     Optional[str] = None
    strategic_priorities:   Optional[List[str]] = None
    industry_associations:  Optional[List[str]] = None
    case_studies:           Optional[List[CaseStudy]] = None
    go_to_market_strategy:  Optional[str] = None
    innovation_roadmap:     Optional[str] = None
    product_pipeline:       Optional[List[str]] = None

    # ── 19. Leadership Roster ─────────────────────────────────────────────
    board_members:          Optional[List[BoardMember]] = None
    marketing_video_url:    Optional[str] = None
    customer_testimonials:  Optional[List[str]] = None
    tech_adoption_rating:   Optional[float] = Field(None, ge=0, le=10)

    # ── 20. Market Sizing ─────────────────────────────────────────────────
    tam:                    Optional[float] = Field(None, ge=0, description="Total Addressable Market in USD")
    sam:                    Optional[float] = Field(None, ge=0, description="Serviceable Addressable Market in USD")
    som:                    Optional[float] = Field(None, ge=0, description="Serviceable Obtainable Market in USD")

    # ── 21. Work Culture ──────────────────────────────────────────────────
    work_culture_summary:       Optional[str] = None
    manager_quality:            Optional[str] = None
    psychological_safety:       Optional[str] = None
    feedback_culture:           Optional[str] = None
    diversity_inclusion_score:  Optional[float] = Field(None, ge=0, le=10)
    ethical_standards:          Optional[str] = None
    typical_hours:              Optional[int] = Field(None, ge=0, le=168)
    overtime_expectations:      Optional[str] = None
    weekend_work:               Optional[str] = None
    flexibility_level:          Optional[FlexibilityEnum] = None
    leave_policy:               Optional[str] = None
    burnout_risk:               Optional[BurnoutRiskEnum] = None

    # ── 22. Location & Infrastructure ─────────────────────────────────────
    location_centrality:        Optional[str] = None
    public_transport_access:    Optional[str] = None
    cab_policy:                 Optional[str] = None
    airport_commute_time:       Optional[str] = None
    office_zone_type:           Optional[str] = None
    area_safety:                Optional[AreaSafetyEnum] = None
    safety_policies:            Optional[List[str]] = None
    infrastructure_safety:      Optional[str] = None
    emergency_preparedness:     Optional[str] = None
    health_support:             Optional[str] = None

    # ── 23. Growth & Development ──────────────────────────────────────────
    onboarding_quality:         Optional[str] = None
    learning_culture:           Optional[str] = None
    exposure_quality:           Optional[str] = None
    mentorship_availability:    Optional[str] = None
    internal_mobility:          Optional[str] = None
    promotion_clarity:          Optional[str] = None
    tools_access:               Optional[str] = None
    role_clarity:               Optional[str] = None
    early_ownership:            Optional[str] = None
    work_impact:                Optional[str] = None
    execution_thinking_balance: Optional[str] = None
    automation_level:           Optional[str] = None
    cross_functional_exposure:  Optional[str] = None

    # ── 24. Company Standing ──────────────────────────────────────────────
    company_maturity:           Optional[CompanyMaturityEnum] = None
    brand_value:                Optional[str] = None
    client_quality:             Optional[str] = None
    layoff_history:             Optional[str] = None

    # ── 25. Compensation ──────────────────────────────────────────────────
    fixed_vs_variable_pay:      Optional[str] = None
    bonus_predictability:       Optional[str] = None
    esops_incentives:           Optional[str] = None
    family_health_insurance:    Optional[str] = None
    relocation_support:         Optional[str] = None
    lifestyle_benefits:         Optional[str] = None

    # ── 26. Career & Network ──────────────────────────────────────────────
    exit_opportunities:         Optional[str] = None
    skill_relevance:            Optional[str] = None
    external_recognition:       Optional[str] = None
    network_strength:           Optional[str] = None
    global_exposure:            Optional[str] = None

    # ── 27. Mission & Sustainability ──────────────────────────────────────
    mission_clarity:            Optional[str] = None
    sustainability_csr:         Optional[str] = None
    crisis_behavior:            Optional[str] = None

    # ═══════════════════════════════════════════════════════════════════════
    # CROSS-FIELD VALIDATORS
    # ═══════════════════════════════════════════════════════════════════════

    @field_validator("primary_contact_email", "contact_person_email", mode="before")
    @classmethod
    def validate_email(cls, v: Any) -> Optional[str]:
        if v is None:
            return v
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(pattern, str(v)):
            raise ValueError(f"Invalid email address: {v}")
        return str(v)

    @field_validator("website_url", "linkedin_url", "facebook_url",
                     "instagram_url", "ceo_linkedin_url", "marketing_video_url",
                     "logo_url", mode="before")
    @classmethod
    def validate_url(cls, v: Any) -> Optional[str]:
        if v is None:
            return v
        s = str(v).strip()
        if s and not re.match(r"^https?://", s, re.IGNORECASE):
            # Attempt to prepend https
            s = "https://" + s
        return s

    @field_validator("twitter_handle", mode="before")
    @classmethod
    def validate_twitter(cls, v: Any) -> Optional[str]:
        if v is None:
            return v
        s = str(v).strip()
        if s and not s.startswith("@"):
            s = "@" + s
        return s

    @model_validator(mode="after")
    def validate_tam_sam_som(self) -> "CompanySchema":
        if self.tam and self.sam and self.tam < self.sam:
            raise ValueError("TAM must be >= SAM")
        if self.sam and self.som and self.sam < self.som:
            raise ValueError("SAM must be >= SOM")
        return self

    @model_validator(mode="after")
    def validate_cac_ltv(self) -> "CompanySchema":
        if (
            self.customer_acquisition_cost is not None
            and self.customer_lifetime_value is not None
            and self.customer_acquisition_cost > 0
        ):
            expected = self.customer_lifetime_value / self.customer_acquisition_cost
            if self.cac_ltv_ratio is not None:
                diff = abs(self.cac_ltv_ratio - expected)
                if diff > 0.5:
                    # Auto-correct instead of raising
                    self.cac_ltv_ratio = round(expected, 4)
        return self

    @model_validator(mode="after")
    def validate_profitability_consistency(self) -> "CompanySchema":
        if (
            self.profitability_status == ProfitabilityEnum.profitable
            and self.annual_profit is not None
            and self.annual_profit < 0
        ):
            raise ValueError(
                "profitability_status is 'profitable' but annual_profit is negative"
            )
        return self
