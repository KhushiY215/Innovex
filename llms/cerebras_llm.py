# llms/cerebras_llm.py
"""
LLM-3: Cerebras Cloud — llama3.1-8b
Uses the cerebras-cloud-sdk (OpenAI-compatible).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    try:
        from cerebras.cloud.sdk import Cerebras
        return Cerebras(api_key=settings.cerebras_api_key)
    except ImportError:
        # Fallback: use openai-compatible client
        from openai import OpenAI
        return OpenAI(
            base_url=settings.cerebras_base_url,
            api_key=settings.cerebras_api_key,
        )


def call_cerebras(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    """
    Call the Cerebras LLM.

    Returns
    -------
    str — raw text content from the model
    """
    client = _get_client()
    logger.info("Calling Cerebras model: %s", settings.cerebras_model)

    completion = client.chat.completions.create(
        model=settings.cerebras_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response: str = completion.choices[0].message.content or ""
    return response.strip()
