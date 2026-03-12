# graph/workflow.py
"""
LangGraph workflow
──────────────────
Intermediate outputs saved per iteration:
  outputs/<company>_iter<N>_llm_outputs.json   ← Agent 1: all 3 LLM results
  outputs/<company>_iter<N>_consolidated.json  ← Agent 2: consolidated result
  outputs/<company>_iter<N>_test_report.json   ← Agent 3: test failures report
  outputs/<company>.json                        ← Final validated output
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from agents.agent1_extractor import run_agent1
from agents.agent2_consolidator import run_agent2
from agents.agent3_tester import run_agent3
from validation.validator import ValidationResult
from graph.state import AgentState
from config.settings import settings

# NEW imports (safe logging only)
from database.company_utils import generate_company_id
from database.store_llm_outputs import store_llm_output
from database.store_consolidated import store_consolidated_output

logger = logging.getLogger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _safe_name(company: str) -> str:
    return company.replace(" ", "_").replace("/", "-").replace("\\", "-")


def _out_dir() -> Path:
    p = Path(settings.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _vr_to_dict(vr: ValidationResult) -> Dict[str, Any]:
    return {
        "is_valid": vr.is_valid,
        "data": vr.data or {},
        "errors": [{"field": e.field, "message": e.message} for e in vr.errors],
    }


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("💾  Saved → %s", path)


# ─── Node 1 : extract ─────────────────────────────────────────────────────────

def node_extract(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0) + 1
    feedback = state.get("feedback", "")
    company = state["company_name"]

    # ensure company_id exists
    company_id = state.get("company_id")
    if not company_id:
        company_id = generate_company_id(company)
        state["company_id"] = company_id

    logger.info("═══ ITERATION %d — Agent 1 (Extract) ═══", iteration)

    llm_results = run_agent1(company, feedback)

    llm1_d = _vr_to_dict(llm_results["llm1_hf"])
    llm2_d = _vr_to_dict(llm_results["llm2_nvidia"])
    llm3_d = _vr_to_dict(llm_results["llm3_cerebras"])

    # ── NEW: store raw LLM outputs in Supabase ──
    try:
        store_llm_output(company, company_id, "hf_llm", iteration, llm1_d["data"])
        store_llm_output(company, company_id, "nvidia_llm", iteration, llm2_d["data"])
        store_llm_output(company, company_id, "cerebras_llm", iteration, llm3_d["data"])
    except Exception as e:
        logger.warning("[DB] Failed storing LLM outputs: %s", e)

    # ── Save Agent 1 combined file ────────────────────────────────────────
    agent1_payload = {
        "company": company,
        "iteration": iteration,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "description": "Raw outputs from all 3 LLMs before consolidation",
        "llm_outputs": {
            "llm1_hf": {
                "provider": "HuggingFace / Groq-fallback",
                "is_valid": llm1_d["is_valid"],
                "validation_errors": llm1_d["errors"],
                "data": llm1_d["data"],
            },
            "llm2_nvidia": {
                "provider": "NVIDIA NIM",
                "is_valid": llm2_d["is_valid"],
                "validation_errors": llm2_d["errors"],
                "data": llm2_d["data"],
            },
            "llm3_cerebras": {
                "provider": "Cerebras",
                "is_valid": llm3_d["is_valid"],
                "validation_errors": llm3_d["errors"],
                "data": llm3_d["data"],
            },
        },
        "summary": {
            "total_llms": 3,
            "fully_valid": sum(1 for d in (llm1_d, llm2_d, llm3_d) if d["is_valid"]),
            "partial_data": sum(1 for d in (llm1_d, llm2_d, llm3_d) if d["data"] and not d["is_valid"]),
            "hard_failed": sum(1 for d in (llm1_d, llm2_d, llm3_d) if not d["data"]),
        },
    }

    out_path = _out_dir() / f"{_safe_name(company)}_iter{iteration}_llm_outputs.json"
    _save_json(out_path, agent1_payload)

    return {
        **state,
        "iteration": iteration,
        "llm1_result": llm1_d,
        "llm2_result": llm2_d,
        "llm3_result": llm3_d,
        "llm_results_valid": any(d["data"] for d in (llm1_d, llm2_d, llm3_d)),
        "_llm_vr_map": llm_results,
        "agent1_output_path": str(out_path),
    }


# ─── Node 2 : consolidate ─────────────────────────────────────────────────────

def node_consolidate(state: AgentState) -> AgentState:
    iteration = state["iteration"]
    company = state["company_name"]
    company_id = state.get("company_id")

    logger.info("═══ ITERATION %d — Agent 2 (Consolidate) ═══", iteration)

    llm_vr_map = state.get("_llm_vr_map", {})
    if not llm_vr_map:
        from validation.validator import ValidationResult as VR

        def _d2vr(d: Dict) -> VR:
            return VR(is_valid=d["is_valid"], data=d.get("data") or {})

        llm_vr_map = {
            "llm1_hf": _d2vr(state["llm1_result"]),
            "llm2_nvidia": _d2vr(state["llm2_result"]),
            "llm3_cerebras": _d2vr(state["llm3_result"]),
        }

    vr = run_agent2(company, llm_vr_map)

    # ── NEW: store consolidated result ──
    try:
        store_consolidated_output(
            company,
            company_id,
            iteration,
            vr.data or {},
            vr.is_valid
        )
    except Exception as e:
        logger.warning("[DB] Failed storing consolidated output: %s", e)

    # ── Save Agent 2 consolidated file ────────────────────────────────────
    agent2_payload = {
        "company": company,
        "iteration": iteration,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "description": "Best single JSON produced by the consolidation judge",
        "judge_model": "Groq",
        "is_valid": vr.is_valid,
        "validation_errors": [{"field": e.field, "message": e.message} for e in vr.errors],
        "consolidated_data": vr.data or {},
    }

    out_path = _out_dir() / f"{_safe_name(company)}_iter{iteration}_consolidated.json"
    _save_json(out_path, agent2_payload)

    return {
        **state,
        "consolidated": vr.data or {},
        "consolidated_valid": vr.is_valid,
        "_consolidated_vr": vr,
        "agent2_output_path": str(out_path),
    }


# ─── Node 3 : test ────────────────────────────────────────────────────────────

def node_test(state: AgentState) -> AgentState:
    iteration = state["iteration"]
    company = state["company_name"]
    logger.info("═══ ITERATION %d — Agent 3 (Test) ═══", iteration)

    consolidated = state.get("consolidated", {})
    passed, feedback, failures, test_summary = run_agent3(consolidated)

    state["test_passed"] = passed
    state["test_feedback"] = feedback
    state["test_failures"] = failures
    state["test_summary"] = test_summary

    agent3_payload = {
        "company": company,
        "iteration": iteration,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "test_summary": test_summary,
        "all_tests_passed": passed,
        "total_failures": len(failures),
        "failures": failures,
        "correction_feedback": feedback if not passed else "N/A",
    }

    out_path = _out_dir() / f"{_safe_name(company)}_iter{iteration}_test_report.json"
    _save_json(out_path, agent3_payload)

    return {
        **state,
        "test_passed": passed,
        "test_failures": failures,
        "feedback": feedback,
        "agent3_output_path": str(out_path),
    }


# ─── Node 4 : save final ──────────────────────────────────────────────────────

def node_save(state: AgentState) -> AgentState:
    company = state["company_name"]
    data = state["consolidated"]

    out_path = _out_dir() / f"{_safe_name(company)}.json"
    _save_json(out_path, data)
    logger.info("✅  Final output saved → %s", out_path)

    return {
        **state,
        "done": True,
        "output_path": str(out_path),
    }


# ─── routing ──────────────────────────────────────────────────────────────────

def route_after_test(state: AgentState) -> str:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", settings.max_iterations)

    if state.get("test_passed"):
        logger.info("All tests passed ✅  → saving final output")
        return "save"

    if iteration >= max_iter:
        logger.warning("Max iterations reached — saving best available output")
        return "save"

    logger.info("Tests failed — looping back to Agent 1")
    return "extract"


# ─── build & run ──────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("extract", node_extract)
    builder.add_node("consolidate", node_consolidate)
    builder.add_node("test", node_test)
    builder.add_node("save", node_save)

    builder.set_entry_point("extract")
    builder.add_edge("extract", "consolidate")
    builder.add_edge("consolidate", "test")
    builder.add_conditional_edges(
        "test", route_after_test,
        {"save": "save", "extract": "extract"},
    )
    builder.add_edge("save", END)

    return builder.compile()


def run_pipeline(
    company_name: str,
    max_iterations: int | None = None,
) -> Dict[str, Any]:

    graph = build_workflow()

    initial_state: AgentState = {
        "company_name": company_name,
        "company_id": generate_company_id(company_name),
        "iteration": 0,
        "max_iterations": max_iterations or settings.max_iterations,
        "feedback": "",
        "done": False,
        "output_path": None,
    }

    return graph.invoke(initial_state)