from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json

from graph.workflow import run_pipeline
from config.settings import settings


app = FastAPI(
    title="Company Intelligence Agent",
    version="1.0"
)


# -------------------------
# Request schema
# -------------------------

class CompanyRequest(BaseModel):
    company_name: str
    max_iterations: int = 3


# -------------------------
# Root
# -------------------------

@app.get("/")
def root():
    return {"message": "Company Intelligence API running"}


# -------------------------
# Run full agent pipeline
# -------------------------

@app.post("/analyze")
def analyze_company(req: CompanyRequest):

    result = run_pipeline(
        company_name=req.company_name,
        max_iterations=req.max_iterations
    )

    output_path = result.get("output_path")

    if not output_path:
        raise HTTPException(500, "Pipeline failed")

    return {
        "status": "completed",
        "iterations": result.get("iteration"),
        "tests_passed": result.get("test_passed"),
        "output": json.loads(Path(output_path).read_text())
    }


# -------------------------
# Get stored company data
# -------------------------

@app.get("/company/{company_name}")
def get_company(company_name: str):

    safe = company_name.replace(" ", "_")
    path = Path(settings.output_dir) / f"{safe}.json"

    if not path.exists():
        raise HTTPException(404, "Company not analyzed yet")

    return json.loads(path.read_text())


# -------------------------
# Get iteration debug info
# -------------------------

@app.get("/runs/{company_name}")
def get_runs(company_name: str):

    safe = company_name.replace(" ", "_")
    out_dir = Path(settings.output_dir)

    files = sorted(out_dir.glob(f"{safe}_iter*"))

    return {
        "company": company_name,
        "runs": [str(f) for f in files]
    }