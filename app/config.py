from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_MODEL_BY_VARIANT: dict[str, str] = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # --- Engine (QWEN3_* prefix) ---------------------------------------------
    qwen3_model: Optional[str] = Field(
        default=None,
        description="HF repo id or local path; defaults per-variant when unset.",
    )
    qwen3_variant: Literal["base", "customvoice", "voicedesign"] = Field(
        default="base",
        description="Model variant selector.",
    )
    qwen3_device: Literal["auto", "cuda", "cpu"] = "auto"
    qwen3_cuda_index: int = Field(default=0, ge=0)
    qwen3_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    qwen3_attn_impl: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"
    qwen3_prompt_cache_size: int = Field(default=16, ge=1)

    # --- Service-level (no prefix) -------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    voices_dir: str = "/voices"
    max_input_chars: int = Field(default=8000, ge=1)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = "mp3"
    max_concurrency: int = Field(default=1, ge=1)
    max_queue_size: int = Field(default=0, ge=0)
    queue_timeout: float = Field(default=0.0, ge=0.0)
    max_audio_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    cors_enabled: bool = False

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir)

    @property
    def effective_model(self) -> str:
        if self.qwen3_model:
            return self.qwen3_model
        return _DEFAULT_MODEL_BY_VARIANT[self.qwen3_variant]

    @property
    def resolved_device(self) -> str:
        if self.qwen3_device == "cpu":
            return "cpu"
        if self.qwen3_device == "cuda":
            return f"cuda:{self.qwen3_cuda_index}"
        # auto
        import torch

        if torch.cuda.is_available():
            return f"cuda:{self.qwen3_cuda_index}"
        return "cpu"

    @property
    def torch_dtype(self):
        import torch

        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.qwen3_dtype]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
