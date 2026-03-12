# agents/agent1_extractor.py
"""
Agent 1 — Company Data Extractor
─────────────────────────────────
• Sends company name to 3 LLMs in parallel (ThreadPoolExecutor)
• Extracts all 163 parameters from each
• Validates each output with Pydantic (schema, types, enums, cross-fields)
• Returns validated dict per LLM
• FAULT-TOLERANT: any LLM that fails (auth, timeout, parse error, etc.)
  is skipped gracefully; the pipeline continues with however many LLMs
  succeeded.  Only raises if ALL 3 LLMs returned zero data whatsoever.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from llms import call_huggingface, call_nvidia, call_cerebras
from prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from validation.validator import validate_company_data, ValidationResult, FieldError

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """Strip markdown fences and parse first JSON object found."""
    m = _JSON_FENCE.search(raw)
    text = m.group(1) if m else raw.strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start : end + 1])


def _make_error_result(field: str, message: str) -> ValidationResult:
    """Create a completely-empty ValidationResult for hard failures."""
    return ValidationResult(
        is_valid=False,
        data=None,                  # None signals: LLM returned nothing usable
        errors=[FieldError(field=field, message=message)],
    )


# ─── per-LLM worker ───────────────────────────────────────────────────────────

def _call_and_validate(
    llm_name: str,
    caller,
    company_name: str,
    feedback: str,
) -> Dict:
    """
    Call one LLM, parse JSON, run Pydantic validation.
    ALL exceptions are caught — this function never raises.

    Return dict keys:
        llm      : str  — LLM name
        result   : ValidationResult
        hard_fail: bool — True only when the LLM call itself failed (no data at all)
    """
    user_prompt = build_extraction_user_prompt(company_name, feedback)
    logger.info("[Agent1] Calling %s …", llm_name)

    # ── Step 1: LLM call ─────────────────────────────────────────────────
    try:
        raw = caller(EXTRACTION_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        short = str(exc)[:200]
        logger.warning("[Agent1] ⚠️  %s SKIPPED — LLM call failed: %s", llm_name, short)
        logger.debug("[Agent1] %s traceback:\n%s", llm_name, traceback.format_exc())
        return {
            "llm":       llm_name,
            "result":    _make_error_result("__llm_call__", short),
            "hard_fail": True,
        }

    logger.debug("[Agent1] %s raw (first 300 chars): %s", llm_name, raw[:300])

    # ── Step 2: JSON parse ───────────────────────────────────────────────
    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[Agent1] ⚠️  %s JSON parse error: %s", llm_name, exc)
        return {
            "llm":       llm_name,
            "result":    _make_error_result("__parse__", str(exc)),
            "hard_fail": True,   # no dict at all
        }

    # ── Step 3: Pydantic validation ──────────────────────────────────────
    #   validate_company_data always sets result.data = raw dict even on failure,
    #   so partial data is preserved for Agent 2.
    try:
        result = validate_company_data(data)
    except Exception as exc:
        logger.warning("[Agent1] ⚠️  %s Pydantic raised unexpectedly: %s", llm_name, exc)
        # Still keep the raw dict
        result = ValidationResult(
            is_valid=False,
            data=data,
            errors=[FieldError(field="__validation__", message=str(exc))],
        )

    status = (
        "✅ VALID"
        if result.is_valid
        else f"⚠️  PARTIAL ({len(result.errors)} schema errors — data kept for consolidation)"
    )
    logger.info("[Agent1] %s → %s", llm_name, status)
    return {"llm": llm_name, "result": result, "hard_fail": False}


# ─── main entry ───────────────────────────────────────────────────────────────

def run_agent1(
    company_name: str,
    feedback: str = "",
) -> Dict[str, ValidationResult]:
    """
    Run Agent 1 — query all 3 LLMs concurrently, validate each output.

    Fault-tolerance
    ───────────────
    • Any LLM that throws (auth, network, gated-repo, timeout) → skipped,
      its result has is_valid=False and data=None.
    • Any LLM that returns malformed JSON → skipped similarly.
    • Any LLM with Pydantic errors → data is preserved as partial dict,
      is_valid=False, errors list the bad fields.
    • Pipeline only raises RuntimeError when ALL 3 LLMs returned data=None
      (i.e. zero information is available).

    Returns
    -------
    {
      "llm1_hf":       ValidationResult,
      "llm2_nvidia":   ValidationResult,
      "llm3_cerebras": ValidationResult,
    }
    """
    tasks = [
        ("llm1_hf",       call_huggingface),
        ("llm2_nvidia",   call_nvidia),
        ("llm3_cerebras", call_cerebras),
    ]

    results: Dict[str, ValidationResult] = {}
    hard_fails: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_call_and_validate, name, caller, company_name, feedback): name
            for name, caller in tasks
        }
        for future in as_completed(futures):
            try:
                payload = future.result()
            except Exception as exc:
                llm_name = futures[future]
                logger.error("[Agent1] Unhandled future error for %s: %s", llm_name, exc)
                results[llm_name] = _make_error_result("__future__", str(exc))
                hard_fails.append(llm_name)
                continue

            results[payload["llm"]] = payload["result"]
            if payload.get("hard_fail"):
                hard_fails.append(payload["llm"])

    # ── Summary log ───────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("[Agent1] Results summary:")
    for llm_name, vr in results.items():
        if vr.is_valid:
            status = "✅ VALID (all 163 fields clean)"
        elif vr.data:
            status = f"⚠️  PARTIAL ({len(vr.errors)} errors, data forwarded to Agent 2)"
        else:
            status = "❌ HARD FAIL (no data — skipped)"
        logger.info("  %-18s → %s", llm_name, status)

    if hard_fails:
        logger.warning(
            "[Agent1] %d LLM(s) hard-failed (no data): %s — continuing with the rest.",
            len(hard_fails), hard_fails,
        )

    # ── Only abort when truly zero data across all LLMs ──────────────────
    has_any_data = any(vr.data for vr in results.values())
    if not has_any_data:
        raise RuntimeError(
            "[Agent1] ALL 3 LLMs failed to return any data. "
            "Check API keys, network, and model access.\n"
            + "\n".join(
                f"  {n}: {results[n].errors[0].message if results[n].errors else 'unknown'}"
                for n in results
            )
        )

    return results