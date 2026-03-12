# llms/huggingface_llm.py

"""
LLM-1 via HuggingFace Inference API.

This replaces the local transformers pipeline to avoid RAM crashes.
If HF API fails, it falls back to Groq (llama-3.1-8b-instant).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# HUGGINGFACE API
# ─────────────────────────────────────────────────────────

def _call_huggingface_api(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    try:
        from huggingface_hub import InferenceClient

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        if not hf_token:
            raise RuntimeError("HF_TOKEN missing")

        client = InferenceClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            token=hf_token,
        )

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )

        return (completion.choices[0].message.content or "").strip()

    except Exception as exc:

        logger.warning(
            "HuggingFace API failed (%s) — switching to Groq fallback",
            str(exc)[:200],
        )

        return _call_groq_fallback(system_prompt, user_prompt, max_tokens)


# ─────────────────────────────────────────────────────────
# GROQ FALLBACK
# ─────────────────────────────────────────────────────────

def _call_groq_fallback(system_prompt: str, user_prompt: str, max_tokens: int) -> str:

    logger.info("[HF-Fallback] Using Groq llama-3.1-8b-instant")

    from groq import Groq
    from config.settings import settings

    client = Groq(api_key=settings.groq_api_key)

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )

    return (completion.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────
# PUBLIC API (UNCHANGED)
# ─────────────────────────────────────────────────────────

def call_huggingface(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 4096,
) -> str:
    """
    Call LLM-1.

    Priority:
    1. HuggingFace API
    2. Groq fallback

    This keeps the same interface used by your agents.
    """

    return _call_huggingface_api(system_prompt, user_prompt, max_new_tokens)