# validation/validator.py
"""
Reusable validation layer — used by Agent 1, Agent 2, and Agent 3.

validate_company_data(raw: dict) -> ValidationResult
  • Schema correctness (all fields exist & typed correctly)
  • Required field checks
  • Data-type checks
  • Enum validation
  • Cross-field consistency

KEY DESIGN: On Pydantic validation failure the RAW input dict is always
preserved in result.data so downstream agents can still use the partial
data.  Fields that failed validation are recorded in result.errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from schema.company_schema import CompanySchema


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldError:
    field:   str
    message: str
    value:   Any = None


@dataclass
class ValidationResult:
    is_valid:   bool
    data:       Optional[Dict[str, Any]] = None   # always set when we have ANY dict from the LLM
    schema_obj: Optional[CompanySchema]  = None   # only set when fully valid
    errors:     List[FieldError]         = field(default_factory=list)
    warnings:   List[str]                = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            return "✅  Validation PASSED"
        lines = [f"❌  Validation FAILED — {len(self.errors)} error(s)"]
        for e in self.errors:
            lines.append(f"   • [{e.field}] {e.message}  (got: {e.value!r})")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────

def _coerce_raw(raw: Any) -> Dict[str, Any]:
    """Accept dict, JSON string, or string-wrapped JSON."""
    if isinstance(raw, CompanySchema):
        return raw.model_dump()
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict/JSON, got {type(raw)}")
    return raw


def validate_company_data(raw: Any, strict: bool = False) -> ValidationResult:
    """
    Validate company data at any pipeline stage.

    Parameters
    ----------
    raw    : dict | str | CompanySchema — the data to validate
    strict : if True, null optional fields are reported as warnings

    Returns
    -------
    ValidationResult
      • is_valid=True  → data is the Pydantic-cleaned dict, schema_obj is set
      • is_valid=False → data is STILL the raw input dict (partial but usable),
                         errors lists every field that failed
    """
    # ── 1. Parse input into a plain dict ─────────────────────────────────
    try:
        raw_dict = _coerce_raw(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        # Cannot even get a dict — truly empty result
        return ValidationResult(
            is_valid=False,
            data=None,
            errors=[FieldError(field="__root__", message=str(exc))],
        )

    # ── 2. Pydantic validation ────────────────────────────────────────────
    try:
        model = CompanySchema(**raw_dict)
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(
                FieldError(
                    field=loc,
                    message=err["msg"],
                    value=err.get("input"),
                )
            )
        # *** KEY FIX: preserve raw_dict so downstream agents can still use it ***
        return ValidationResult(
            is_valid=False,
            data=raw_dict,      # partial data — always populated when LLM returned a dict
            errors=errors,
        )

    # ── 3. Success path ───────────────────────────────────────────────────
    warnings: List[str] = []
    if strict:
        for fname in model.model_fields:
            if getattr(model, fname) is None:
                warnings.append(f"Optional field '{fname}' is null/missing")

    cleaned = model.model_dump(mode="json")
    return ValidationResult(
        is_valid=True,
        data=cleaned,
        schema_obj=model,
        warnings=warnings,
    )


def validate_many(
    outputs: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, ValidationResult]:
    """Validate a dict of {llm_name: raw_output}. Returns {llm_name: ValidationResult}."""
    return {name: validate_company_data(raw, strict) for name, raw in outputs.items()}


def extract_valid_or_raise(raw: Any, context: str = "") -> Dict[str, Any]:
    """Validate and return cleaned dict, raising RuntimeError on failure."""
    result = validate_company_data(raw)
    if not result.is_valid:
        prefix = f"[{context}] " if context else ""
        raise RuntimeError(prefix + result.summary())
    return result.data