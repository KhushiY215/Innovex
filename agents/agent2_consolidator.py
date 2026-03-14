"""
Agent 2 — Consolidation Judge
──────────────────────────────
• Accepts the 3 validated JSON outputs from Agent 1
• Sends them all to Groq (llama-3.3-70b-versatile) for consolidation
• Normalizes known schema issues
• Validates the consolidated output with Pydantic
• Returns final validated dict
• FAULT-TOLERANT: works even if 1 or 2 LLMs were skipped
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

from llms import call_groq_consolidator
from prompts import CONSOLIDATION_SYSTEM_PROMPT, build_consolidation_user_prompt
from validation.validator import validate_company_data, ValidationResult, FieldError

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ------------------------------------------------
# JSON extraction helper
# ------------------------------------------------
def _extract_json(raw: str) -> dict:
    m = _JSON_FENCE.search(raw)
    text = m.group(1) if m else raw.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in consolidation response")

    return json.loads(text[start : end + 1])


# ------------------------------------------------
# Normalization layer (NEW)
# Fixes common LLM mistakes before validation
# ------------------------------------------------
def _normalize_fields(data: dict) -> dict:
    
    if not isinstance(data, dict):
        return data

    # ------------------------------------------
    # diversity_inclusion_score must be 0–10
    # ------------------------------------------
    if "diversity_inclusion_score" in data:
        score = data.get("diversity_inclusion_score")

        try:
            score = float(score)

            if score > 10:
                data["diversity_inclusion_score"] = score / 10

        except Exception:
            pass


    # ------------------------------------------
    # ESG rating normalization
    # ------------------------------------------
    if "esg_ratings" in data and isinstance(data["esg_ratings"], dict):

        mapping = {
            "strong": "high",
            "very strong": "exemplary",
            "average": "medium",
            "weak": "low"
        }

        for key, val in data["esg_ratings"].items():

            if isinstance(val, str):
                val_lower = val.lower()

                if val_lower in mapping:
                    data["esg_ratings"][key] = mapping[val_lower]


    # ------------------------------------------
    # CAC/LTV ratio correction
    # ------------------------------------------
    cac = data.get("customer_acquisition_cost")
    ltv = data.get("customer_lifetime_value")

    try:
        if cac and ltv and float(cac) > 0:
            data["cac_ltv_ratio"] = float(ltv) / float(cac)
    except Exception:
        pass


    return data


# ------------------------------------------------
# Best-effort JSON from agent1
# ------------------------------------------------
def _best_effort_json(vr: Optional[ValidationResult], llm_label: str) -> str:

    if vr is None:
        return json.dumps({"_skipped": True, "_reason": "LLM not in results"}, indent=2)

    if vr.is_valid and vr.data:
        return json.dumps(vr.data, indent=2)

    if vr.data:
        logger.warning(
            "[Agent2] %s has %d validation errors — passing partial data",
            llm_label,
            len(vr.errors),
        )
        return json.dumps(vr.data, indent=2)

    reason = vr.errors[0].message if vr.errors else "unknown"

    logger.warning(
        "[Agent2] %s produced no usable data (%s) — sending placeholder",
        llm_label,
        reason[:120],
    )

    return json.dumps({"_skipped": True, "_reason": reason[:200]}, indent=2)


# ------------------------------------------------
# Main Agent2 function
# ------------------------------------------------
def run_agent2(
    company_name: str,
    llm_results: Dict[str, ValidationResult],
) -> ValidationResult:

    llm1_json = _best_effort_json(llm_results.get("llm1_hf"), "llm1_hf")
    llm2_json = _best_effort_json(llm_results.get("llm2_nvidia"), "llm2_nvidia")
    llm3_json = _best_effort_json(llm_results.get("llm3_cerebras"), "llm3_cerebras")

    usable_count = sum(
        1 for j in (llm1_json, llm2_json, llm3_json) if '"_skipped": true' not in j
    )

    logger.info("[Agent2] Consolidating from %d/3 usable LLM outputs …", usable_count)

    user_prompt = build_consolidation_user_prompt(
        llm1_json, llm2_json, llm3_json, company_name
    )

    logger.info("[Agent2] Calling Groq consolidation judge …")

    try:
        raw = call_groq_consolidator(CONSOLIDATION_SYSTEM_PROMPT, user_prompt)

    except Exception as exc:
        logger.error("[Agent2] Groq call failed: %s", exc)

        return ValidationResult(
            is_valid=False,
            errors=[FieldError(field="__agent2_call__", message=str(exc))],
        )

    logger.debug("[Agent2] Groq raw response (first 300 chars): %s", raw[:300])

    # ----------------------------
    # Parse JSON
    # ----------------------------
    try:
        data = _extract_json(raw)

    except (ValueError, json.JSONDecodeError) as exc:

        logger.error("[Agent2] JSON parse error: %s", exc)

        return ValidationResult(
            is_valid=False,
            errors=[FieldError(field="__parse__", message=str(exc))],
        )

    # ----------------------------
    # NEW: normalize fields
    # ----------------------------
    data = _normalize_fields(data)

    # ----------------------------
    # Validate schema
    # ----------------------------
    result = validate_company_data(data)

    logger.info(
        "[Agent2] Consolidation validation: %s",
        "✅ PASSED" if result.is_valid else f"⚠️  {len(result.errors)} errors",
    )

    # Keep raw data even if validation fails
    if not result.is_valid and not result.data and data:
        result.data = data

    return result