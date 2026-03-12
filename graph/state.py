# graph/state.py
"""
LangGraph typed state shared across all nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Input
    company_name:       str

    # Agent 1 outputs (per-LLM ValidationResult serialised as dicts)
    llm1_result:        Dict[str, Any]   # from HuggingFace
    llm2_result:        Dict[str, Any]   # from NVIDIA
    llm3_result:        Dict[str, Any]   # from Cerebras
    llm_results_valid:  bool             # True if at least 2/3 valid

    # Agent 2 output
    consolidated:       Dict[str, Any]   # cleaned consolidated dict
    consolidated_valid: bool

    # Agent 3 output
    test_passed:        bool
    test_failures:      List[Dict[str, Any]]
    feedback:           str              # correction text → fed back to Agent 1

    # Loop control
    iteration:          int
    max_iterations:     int
    done:               bool

    # Final output path
    output_path:        Optional[str]
