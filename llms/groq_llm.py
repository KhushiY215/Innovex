# llms/groq_llm.py
"""
LLM-4 / LLM-5: Groq Cloud
  • groq_model       = llama-3.3-70b-versatile   (Agent 2 — consolidation judge)
  • groq_judge_model = qwen/qwen3-32b             (Agent 3 — test feedback analyst)
"""

from __future__ import annotations

import logging
from functools import lru_cache

from groq import Groq

from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def _call(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8000,
    temperature: float = 0.1,
) -> str:
    client = _get_client()
    logger.info("Calling Groq model: %s", model)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def call_groq_consolidator(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8000,
) -> str:
    """Agent 2 judge — llama-3.3-70b-versatile."""
    return _call(settings.groq_model, system_prompt, user_prompt, max_tokens)


def call_groq_analyst(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> str:
    """Agent 3 analyst — qwen/qwen3-32b."""
    return _call(settings.groq_judge_model, system_prompt, user_prompt, max_tokens)
