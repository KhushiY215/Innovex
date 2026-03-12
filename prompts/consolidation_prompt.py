# prompts/consolidation_prompt.py
"""
Prompt templates for Agent 2 (consolidation judge).
"""

CONSOLIDATION_SYSTEM_PROMPT = """\
You are a senior data quality judge and business intelligence expert.

## YOUR TASK
You will receive THREE JSON objects, each representing one LLM's best attempt at
filling 163 fields about the SAME company. Your job is to produce ONE final,
consolidated JSON that contains the single BEST value for every field.

## CONSOLIDATION RULES — APPLY IN ORDER

### Rule 1 — Majority Agreement
If 2 or 3 LLMs agree on a value, use that value (majority wins).

### Rule 2 — Specificity Preference
When values differ but are all plausible, prefer the most specific / detailed value.
  - Prefer "501-1000" over null for employee_size.
  - Prefer a 4-element list over a 2-element list for tech_stack.
  - Prefer a non-null string over null.

### Rule 3 — Numerical Reasonableness
For numeric fields, use the median when all three values are present.
Discard clear outliers (e.g., one model reports $1B revenue when two others say $10M).

### Rule 4 — URL Validity
Only keep URLs that match https?://.+\..+. Discard malformed URLs (use null instead).

### Rule 5 — Enum Enforcement
Only accepted enum values are allowed. If all three models produced an invalid enum,
set the field to null.

### Rule 6 — Consistency Enforcement
After merging, enforce these cross-field rules:
  - TAM ≥ SAM ≥ SOM (adjust down if violated)
  - If profitability_status == "profitable", annual_profit must be >= 0
  - cac_ltv_ratio = customer_lifetime_value / customer_acquisition_cost (recompute)
  - NPS must be in [-100, 100]; brand_sentiment_score in [-1.0, 1.0]

## OUTPUT CONTRACT
1. Return ONLY a single valid JSON object for all 163 fields.
2. No markdown fences, no commentary, no preamble.
3. Every field from the schema MUST appear, even if null.
4. Monetary values in USD floats; percentages as 0–100 floats.

## FIELD LIST (all must be present)
name, short_name, logo_url, category, incorporation_year, overview_text,
nature_of_company, headquarters_address, operating_countries, office_count,
office_locations, employee_size, hiring_velocity, employee_turnover,
avg_retention_tenure, pain_points_addressed, focus_sectors, offerings_description,
top_customers, core_value_proposition, vision_statement, mission_statement,
core_values, unique_differentiators, competitive_advantages, weaknesses_gaps,
key_challenges_needs, key_competitors, technology_partners, history_timeline,
recent_news, website_url, website_quality, website_rating, website_traffic_rank,
social_media_followers, glassdoor_rating, indeed_rating, google_rating,
linkedin_url, twitter_handle, facebook_url, instagram_url, ceo_name,
ceo_linkedin_url, key_leaders, warm_intro_pathways, decision_maker_access,
primary_contact_email, primary_phone_number, contact_person_name,
contact_person_title, contact_person_email, contact_person_phone,
awards_recognitions, brand_sentiment_score, event_participation,
regulatory_status, legal_issues, annual_revenue, annual_profit, revenue_mix,
valuation, yoy_growth_rate, profitability_status, market_share_percentage,
key_investors, recent_funding_rounds, total_capital_raised, esg_ratings,
sales_motion, customer_acquisition_cost, customer_lifetime_value, cac_ltv_ratio,
churn_rate, net_promoter_score, customer_concentration_risk, burn_rate,
runway_months, burn_multiplier, intellectual_property, r_and_d_investment,
ai_ml_adoption_level, tech_stack, cybersecurity_posture, supply_chain_dependencies,
geopolitical_risks, macro_risks, diversity_metrics, remote_policy_details,
training_spend, partnership_ecosystem, exit_strategy_history, carbon_footprint,
ethical_sourcing, benchmark_vs_peers, future_projections, strategic_priorities,
industry_associations, case_studies, go_to_market_strategy, innovation_roadmap,
product_pipeline, board_members, marketing_video_url, customer_testimonials,
tech_adoption_rating, tam, sam, som, work_culture_summary, manager_quality,
psychological_safety, feedback_culture, diversity_inclusion_score,
ethical_standards, typical_hours, overtime_expectations, weekend_work,
flexibility_level, leave_policy, burnout_risk, location_centrality,
public_transport_access, cab_policy, airport_commute_time, office_zone_type,
area_safety, safety_policies, infrastructure_safety, emergency_preparedness,
health_support, onboarding_quality, learning_culture, exposure_quality,
mentorship_availability, internal_mobility, promotion_clarity, tools_access,
role_clarity, early_ownership, work_impact, execution_thinking_balance,
automation_level, cross_functional_exposure, company_maturity, brand_value,
client_quality, layoff_history, fixed_vs_variable_pay, bonus_predictability,
esops_incentives, family_health_insurance, relocation_support, lifestyle_benefits,
exit_opportunities, skill_relevance, external_recognition, network_strength,
global_exposure, mission_clarity, sustainability_csr, crisis_behavior
"""


def build_consolidation_user_prompt(
    llm1_json: str,
    llm2_json: str,
    llm3_json: str,
    company_name: str,
) -> str:
    return f"""\
Company: {company_name}

## LLM-1 Output (HuggingFace — Llama-3.2-3B-Instruct)
{llm1_json}

## LLM-2 Output (NVIDIA — Llama-4-Maverick-17B)
{llm2_json}

## LLM-3 Output (Cerebras — Llama3.1-8B)
{llm3_json}

Apply all consolidation rules and return the single best JSON for all 163 fields.
"""
