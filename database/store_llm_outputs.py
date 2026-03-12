from datetime import datetime
from database.supabase_client import supabase


def store_llm_output(
    company_name: str,
    company_id: str,
    llm_name: str,
    iteration: int,
    data: dict
):

    payload = {
        "company_name": company_name,
        "company_id": company_id,
        "llm_name": llm_name,
        "iteration": iteration,
        "data": data,
        "created_at": datetime.utcnow().isoformat()
    }

    try:
        supabase.table("llm_extractions").insert(payload).execute()
    except Exception as e:
        print(f"[DB] Failed storing LLM output: {e}")