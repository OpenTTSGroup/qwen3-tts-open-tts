from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)


# Language key → display name. Keys must match what
# Qwen3-TTS's ``get_supported_languages()`` returns verbatim, otherwise the
# client cannot pass them back into /v1/audio/speech.
_LANGUAGE_NAMES: dict[str, str] = {
    "Auto": "Auto-detect",
    "Chinese": "Chinese",
    "English": "English",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "German": "German",
    "French": "French",
    "Russian": "Russian",
    "Portuguese": "Portuguese",
    "Spanish": "Spanish",
    "Italian": "Italian",
}

# Synthetic voice id exposed in voicedesign variant so /v1/audio/speech
# remains spec-compliant (voice field is mandatory).
DESIGN_VOICE_ID = "design"

# Qwen3-TTS 12Hz tokenizer output sample rate.
SAMPLE_RATE_HZ = 24000


def _prep_sys_path() -> None:
    engine_root = os.environ.get("QWEN3_ROOT")
    if not engine_root:
        here = os.path.abspath(os.path.dirname(__file__))
        candidate = os.path.abspath(os.path.join(here, "..", "engine"))
        if os.path.isdir(candidate):
            engine_root = candidate
    if engine_root and engine_root not in sys.path:
        sys.path.insert(0, engine_root)


class TTSEngine:
    """Thin wrapper around Qwen3-TTS' ``Qwen3TTSModel``."""

    def __init__(self, settings: Settings) -> None:
        _prep_sys_path()
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        self._settings = settings
        self._variant = settings.qwen3_variant
        self._model_id = settings.effective_model

        log.info(
            "loading Qwen3-TTS model_id=%s variant=%s device=%s dtype=%s attn=%s",
            self._model_id,
            self._variant,
            settings.resolved_device,
            settings.qwen3_dtype,
            settings.qwen3_attn_impl,
        )

        self._model = Qwen3TTSModel.from_pretrained(
            self._model_id,
            device_map=settings.resolved_device,
            dtype=settings.torch_dtype,
            attn_implementation=settings.qwen3_attn_impl,
        )

        # Builtin voices depend on the variant.
        if self._variant == "customvoice":
            supported = self._model.get_supported_speakers() or []
            self._builtin_voices: list[str] = sorted(supported)
        elif self._variant == "voicedesign":
            self._builtin_voices = [DESIGN_VOICE_ID]
        else:  # base
            self._builtin_voices = []

        # Supported languages (static per-model). ``None`` means the model
        # does not advertise a whitelist; expose the full canonical list then.
        supported_langs = self._model.get_supported_languages()
        if supported_langs:
            self._languages = [
                (k, _LANGUAGE_NAMES.get(k, k)) for k in supported_langs
            ]
        else:
            self._languages = list(_LANGUAGE_NAMES.items())

        # Prompt-item cache for base (zero-shot clone) variant.
        self._prompt_cache: "OrderedDict[tuple[str, float], Any]" = OrderedDict()
        self._prompt_cache_max = settings.qwen3_prompt_cache_size

    # --- Introspection --------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def device(self) -> str:
        return str(self._model.device)

    @property
    def dtype_str(self) -> str:
        return self._settings.qwen3_dtype

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE_HZ

    @property
    def builtin_voices_list(self) -> list[str]:
        return list(self._builtin_voices)

    def has_builtin_voice(self, name: str) -> bool:
        # Qwen3-TTS validates speakers case-insensitively.
        return name.lower() in {v.lower() for v in self._builtin_voices}

    def list_languages(self) -> list[tuple[str, str]]:
        return list(self._languages)

    # --- Generation kwargs ----------------------------------------------------

    @staticmethod
    def _gen_kwargs(**extra: Any) -> dict[str, Any]:
        """Strip None values so the engine can fall back to generate_defaults."""
        return {k: v for k, v in extra.items() if v is not None}

    # --- Prompt cache (base variant) -----------------------------------------

    def _get_or_build_prompt(
        self,
        ref_audio: str,
        ref_text: str,
        ref_mtime: Optional[float],
    ) -> Any:
        if ref_mtime is not None:
            key = (ref_audio, float(ref_mtime))
            cached = self._prompt_cache.get(key)
            if cached is not None:
                self._prompt_cache.move_to_end(key)
                return cached
        items = self._model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        if ref_mtime is not None:
            self._prompt_cache[(ref_audio, float(ref_mtime))] = items
            while len(self._prompt_cache) > self._prompt_cache_max:
                self._prompt_cache.popitem(last=False)
        return items

    # --- Synthesis ------------------------------------------------------------

    async def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio: str,
        ref_text: str,
        ref_mtime: Optional[float] = None,
        instructions: Optional[str] = None,  # noqa: ARG002 - base variant ignores
        speed: float = 1.0,  # noqa: ARG002 - base variant ignores
        language: Optional[str] = None,
        **extra: Any,
    ) -> np.ndarray:
        if self._variant != "base":
            raise RuntimeError(
                f"synthesize_clone requires variant=base (got {self._variant})"
            )

        def _run() -> np.ndarray:
            items = self._get_or_build_prompt(ref_audio, ref_text, ref_mtime)
            wavs, _ = self._model.generate_voice_clone(
                text=text,
                language=language or "Auto",
                voice_clone_prompt=items,
                **self._gen_kwargs(**extra),
            )
            return wavs[0]

        return await asyncio.to_thread(_run)

    async def synthesize_builtin(
        self,
        text: str,
        *,
        voice: str,
        instructions: Optional[str] = None,
        speed: float = 1.0,  # noqa: ARG002 - forwarded implicitly; engine ignores
        language: Optional[str] = None,
        **extra: Any,
    ) -> np.ndarray:
        if self._variant == "customvoice":
            def _run() -> np.ndarray:
                wavs, _ = self._model.generate_custom_voice(
                    text=text,
                    speaker=voice,
                    language=language or "Auto",
                    instruct=instructions if instructions else None,
                    **self._gen_kwargs(**extra),
                )
                return wavs[0]

            return await asyncio.to_thread(_run)

        if self._variant == "voicedesign" and voice == DESIGN_VOICE_ID:
            # Route through voice_design; instructions play the role of instruct.
            def _run() -> np.ndarray:
                wavs, _ = self._model.generate_voice_design(
                    text=text,
                    instruct=instructions or "",
                    language=language or "Auto",
                    **self._gen_kwargs(**extra),
                )
                return wavs[0]

            return await asyncio.to_thread(_run)

        raise RuntimeError(
            f"builtin voice {voice!r} not available for variant={self._variant}"
        )

    async def synthesize_design(
        self,
        text: str,
        *,
        instruct: Optional[str],
        language: Optional[str] = None,
        **extra: Any,
    ) -> np.ndarray:
        if self._variant != "voicedesign":
            raise RuntimeError(
                f"synthesize_design requires variant=voicedesign (got {self._variant})"
            )

        def _run() -> np.ndarray:
            wavs, _ = self._model.generate_voice_design(
                text=text,
                instruct=instruct or "",
                language=language or "Auto",
                **self._gen_kwargs(**extra),
            )
            return wavs[0]

        return await asyncio.to_thread(_run)
