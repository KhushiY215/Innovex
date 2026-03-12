# Company Intelligence Agent

A **multi-agent LangGraph pipeline** that extracts all **163 company parameters** using 3 LLMs in parallel, consolidates via a judge LLM, validates with Pydantic, and iteratively corrects via pytest feedback — until all tests pass.

---

## Architecture

```
Input: Company Name
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  AGENT 1 — Extractor                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ LLM-1        │  │ LLM-2        │  │ LLM-3        │    │
│  │ HuggingFace  │  │ NVIDIA NIM   │  │ Cerebras     │    │
│  │ Llama-3.2-3B │  │ Llama-4-Mav  │  │ Llama3.1-8B  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         └─────────────────┼──────────────────┘            │
│                    Pydantic Validation ×3                  │
└───────────────────────────┬───────────────────────────────┘
                            │ 3 validated JSONs
                            ▼
┌───────────────────────────────────────────────────────────┐
│  AGENT 2 — Consolidation Judge                            │
│  LLM-4: Groq (llama-3.3-70b-versatile)                   │
│  • Majority-vote per field                                │
│  • Best-value selection                                   │
│  • Cross-field consistency enforcement                    │
│  • Pydantic re-validation                                 │
└───────────────────────────┬───────────────────────────────┘
                            │ 1 consolidated JSON
                            ▼
┌───────────────────────────────────────────────────────────┐
│  AGENT 3 — Tester & Feedback Generator                    │
│  pytest (35+ test cases) → failure report                 │
│  LLM-5: Groq (qwen/qwen3-32b) → correction instructions  │
└────────────┬──────────────────────────┬───────────────────┘
             │ FAIL: feedback           │ PASS
             ▼                          ▼
      ← loop to Agent 1          Save  <company>.json
```

---

## LLMs Used

| Role       | Provider    | Model                                  |
|------------|-------------|----------------------------------------|
| LLM-1      | HuggingFace | `meta-llama/Llama-3.2-3B-Instruct`    |
| LLM-2      | NVIDIA NIM  | `meta/llama-4-maverick-17b-128e-instruct` |
| LLM-3      | Cerebras    | `llama3.1-8b`                          |
| LLM-4 Judge| Groq        | `llama-3.3-70b-versatile`              |
| LLM-5 Test | Groq        | `qwen/qwen3-32b`                       |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU Note**: LLM-1 (HuggingFace) runs locally. It requires a GPU for reasonable speed.
> On CPU it will fall back automatically but will be slow.

### 2. Configure API keys

Copy `.env` and verify all keys are set:

```bash
cat .env
```

### 3. Run

```bash
python main.py "Apple Inc"
python main.py "Stripe" --max-iterations 5
python main.py "Infosys" --output-dir my_outputs/
```

Output: `outputs/<company_name>.json`

---

## Project Structure

```
company_intelligence_agent/
├── .env                          # API keys
├── requirements.txt
├── main.py                       # CLI entry point
│
├── config/
│   └── settings.py               # Pydantic settings (env-loaded)
│
├── schema/
│   └── company_schema.py         # 163-param Pydantic model + enums
│
├── validation/
│   └── validator.py              # Reusable validation (used by all agents)
│
├── llms/
│   ├── huggingface_llm.py        # LLM-1
│   ├── nvidia_llm.py             # LLM-2
│   ├── cerebras_llm.py           # LLM-3
│   └── groq_llm.py               # LLM-4 & LLM-5
│
├── prompts/
│   ├── extraction_prompt.py      # Agent 1 prompts (with feedback injection)
│   ├── consolidation_prompt.py   # Agent 2 judge prompts
│   └── test_feedback_prompt.py   # Agent 3 correction prompts
│
├── agents/
│   ├── agent1_extractor.py       # Parallel 3-LLM extraction + validation
│   ├── agent2_consolidator.py    # Groq consolidation judge
│   └── agent3_tester.py          # pytest runner + feedback generator
│
├── graph/
│   ├── state.py                  # LangGraph typed state
│   └── workflow.py               # LangGraph DAG (nodes + edges + loop)
│
├── tests/                        # ← single folder for ALL pytest tests
│   ├── conftest.py               # --company-json fixture
│   └── test_company_parameters.py # 35+ test cases
│
└── outputs/                      # Generated <company>.json files
```

---

## Validation Strategy

Validation is centralised in `validation/validator.py` and reused at **all 3 stages**:

| Stage   | When applied                    | What checked                          |
|---------|---------------------------------|---------------------------------------|
| Agent 1 | After each LLM response         | Schema, types, enums, cross-fields    |
| Agent 2 | After consolidation             | Same — plus consistency re-check     |
| Agent 3 | Before pytest + in pytest       | 35+ business-rule tests               |

---

## Test Cases (35+)

Located in `tests/test_company_parameters.py`:

- **Required fields**: `name`
- **URL format**: `website_url`, `linkedin_url`, `ceo_linkedin_url`, etc.
- **Email format**: `primary_contact_email`, `contact_person_email`
- **Phone format**: `primary_phone_number`
- **Rating ranges**: glassdoor (0–5), website (0–10), NPS (-100 to 100)
- **Financial non-negative**: revenue, valuation, burn_rate, etc.
- **Enum values**: all 14 enum fields
- **List non-empty**: tech_stack, focus_sectors, key_competitors, etc.
- **Cross-field**: TAM≥SAM≥SOM, CAC/LTV ratio, profitability consistency
- **Nested shapes**: key_leaders, board_members, funding_rounds, history_timeline

---

## Loop Behaviour

1. Agent 1 runs → 3 validated JSONs
2. Agent 2 consolidates → 1 validated JSON
3. Agent 3 runs pytest → if all pass, save & exit
4. If tests fail: Agent 3 calls Groq to generate correction instructions
5. Instructions are injected into Agent 1's prompt for the next iteration
6. Repeats until all tests pass **or** `max_iterations` is reached

---

## Output Format

`outputs/Apple_Inc.json` — a fully validated JSON with all 163 fields.
