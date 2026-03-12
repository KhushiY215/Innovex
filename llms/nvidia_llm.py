# llms/nvidia_llm.py
"""
LLM-2: NVIDIA NIM — meta/llama-4-maverick-17b-128e-instruct
Uses the OpenAI-compatible endpoint at https://integrate.api.nvidia.com/v1
"""

from __future__ import annotations

import logging
from functools import lru_cache

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
    )


def call_nvidia(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    """
    Call the NVIDIA NIM endpoint.

    Returns
    -------
    str — raw text content from the model
    """
    client = _get_client()
    logger.info("Calling NVIDIA model: %s", settings.nvidia_model)

    completion = client.chat.completions.create(
        model=settings.nvidia_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response: str = completion.choices[0].message.content or ""
    return response.strip()
