"""
Stores Agent 2 consolidated company data into Supabase.

Table used:
    consolidated_companies

Schema expected:
    id UUID
    company_name TEXT
    company_id BIGINT
    iteration INT
    data JSONB
    tests_passed BOOLEAN
    created_at TIMESTAMP
"""

from datetime import datetime
import logging

from database.supabase_client import supabase

logger = logging.getLogger(__name__)


def store_consolidated_output(
    company_name: str,
    company_id: int,
    iteration: int,
    data: dict,
    tests_passed: bool,
) -> None:
    """
    Insert consolidated company data into Supabase.

    This function is intentionally non-blocking for the agent pipeline.
    Any database failure will NOT interrupt the workflow.
    """

    payload = {
        "company_name": company_name,
        "company_id": company_id,
        "iteration": iteration,
        "data": data or {},
        "tests_passed": tests_passed,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        response = (
            supabase
            .table("consolidated_companies")
            .insert(payload)
            .execute()
        )

        logger.info(
            "[DB] Consolidated output stored for %s (iteration %s)",
            company_name,
            iteration,
        )

        return response

    except Exception as e:
        logger.warning(
            "[DB] Failed storing consolidated output for %s (iteration %s): %s",
            company_name,
            iteration,
            e,
        )
        return None