# config/settings.py
"""
Central configuration — loads from .env and exposes typed settings.
"""

from __future__ import annotations
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────
    nvidia_api_key: str = Field(..., alias="NVIDIA_API_KEY")
    cerebras_api_key: str = Field(..., alias="CEREBRAS_API_KEY")
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")

    # ── Model IDs ─────────────────────────────────────────────────────────
    hf_model_id: str = Field("meta-llama/Llama-3.2-3B-Instruct", alias="HF_MODEL_ID")
    nvidia_model: str = Field("meta/llama-4-maverick-17b-128e-instruct", alias="NVIDIA_MODEL")
    cerebras_model: str = Field("llama3.1-8b", alias="CEREBRAS_MODEL")
    groq_model: str = Field("llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_judge_model: str = Field("qwen/qwen3-32b", alias="GROQ_JUDGE_MODEL")

    # ── Agent Settings ────────────────────────────────────────────────────
    max_iterations: int = Field(5, alias="MAX_ITERATIONS")
    output_dir: str = Field("outputs", alias="OUTPUT_DIR")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ── NVIDIA NIM Base URL ───────────────────────────────────────────────
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # ── Cerebras Base URL ─────────────────────────────────────────────────
    cerebras_base_url: str = "https://api.cerebras.ai/v1"


settings = Settings()
