from __future__ import annotations

import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response

from app.audio import CONTENT_TYPES, encode
from app.concurrency import ConcurrencyLimiter
from app.config import Settings, get_settings
from app.engine import DESIGN_VOICE_ID
from app.schemas import (
    Capabilities,
    DesignRequest,
    HealthResponse,
    Language,
    LanguagesResponse,
    SpeechRequest,
    VoiceInfo,
    VoiceListResponse,
)
from app.voices import FILE_VOICE_PREFIX, Voice, VoiceCatalog

log = logging.getLogger(__name__)


CLONE_AUDIO_EXTS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm"}
)


def _capabilities_for(variant: str) -> Capabilities:
    return Capabilities(
        clone=(variant == "base"),
        streaming=False,
        design=(variant == "voicedesign"),
        languages=True,
        builtin_voices=(variant in ("customvoice", "voicedesign")),
    )


# Resolve variant at import time so router registration knows what to mount.
_settings_boot = get_settings()
CAPABILITIES = _capabilities_for(_settings_boot.qwen3_variant)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    app.state.settings = settings
    app.state.catalog = VoiceCatalog(settings.voices_path)
    app.state.limiter = ConcurrencyLimiter(
        settings.max_concurrency,
        settings.max_queue_size,
        settings.queue_timeout,
    )
    app.state.capabilities = CAPABILITIES
    app.state.engine = None

    from app.engine import TTSEngine

    try:
        engine = TTSEngine(settings)
    except Exception:
        log.exception("failed to load Qwen3-TTS engine")
        raise

    app.state.engine = engine
    log.info(
        "engine ready: model=%s variant=%s device=%s dtype=%s sample_rate=%d "
        "builtin_voices=%d",
        engine.model_id,
        engine.variant,
        engine.device,
        engine.dtype_str,
        engine.sample_rate,
        len(engine.builtin_voices_list),
    )

    yield


app = FastAPI(title="qwen3-tts-open-tts", version="1.0.0", lifespan=lifespan)

if _settings_boot.cors_enabled:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Helpers


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _engine(request: Request):
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="engine loading")
    return engine


def _limiter(request: Request) -> ConcurrencyLimiter:
    return request.app.state.limiter


def _catalog(request: Request) -> VoiceCatalog:
    return request.app.state.catalog


def _resolve_format(fmt: Optional[str], settings: Settings) -> str:
    chosen = fmt or settings.default_response_format
    if chosen not in CONTENT_TYPES:
        raise HTTPException(
            status_code=422, detail=f"unsupported response_format: {chosen}"
        )
    return chosen


def _validate_text(text: str, limit: int) -> None:
    if len(text) == 0:
        raise HTTPException(status_code=422, detail="input must not be empty")
    if len(text) > limit:
        raise HTTPException(status_code=413, detail=f"input exceeds {limit} chars")


def _resolve_voice(voice: str, request: Request) -> tuple[str, Optional[Voice]]:
    if voice.startswith(FILE_VOICE_PREFIX):
        if not CAPABILITIES.clone:
            raise HTTPException(
                status_code=422,
                detail="file:// voices not supported by this variant",
            )
        found = _catalog(request).get(voice)
        if found is None:
            raise HTTPException(status_code=404, detail=f"voice '{voice}' not found")
        return "clone", found

    for scheme in ("http://", "https://", "s3://"):
        if voice.startswith(scheme):
            raise HTTPException(
                status_code=501, detail="remote voice URIs not supported"
            )

    if not CAPABILITIES.builtin_voices:
        raise HTTPException(
            status_code=422,
            detail="voice must use 'file://' prefix for this variant",
        )
    engine = _engine(request)
    if not engine.has_builtin_voice(voice):
        raise HTTPException(status_code=404, detail=f"voice '{voice}' not found")
    return "builtin", None


def _speech_extras(req: SpeechRequest) -> dict[str, object]:
    return {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "repetition_penalty": req.repetition_penalty,
        "max_new_tokens": req.max_new_tokens,
    }


# ---------------------------------------------------------------------------
# Endpoints


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    settings = _settings(request)
    engine = request.app.state.engine
    limiter = _limiter(request)

    if engine is None:
        return HealthResponse(
            status="loading",
            model=settings.effective_model,
            sample_rate=0,
            capabilities=CAPABILITIES,
            variant=settings.qwen3_variant,
            concurrency=limiter.snapshot(),
        )

    return HealthResponse(
        status="ok",
        model=engine.model_id,
        sample_rate=engine.sample_rate,
        capabilities=CAPABILITIES,
        device=engine.device,
        dtype=engine.dtype_str,
        variant=engine.variant,
        concurrency=limiter.snapshot(),
    )


@app.get("/v1/audio/voices", response_model=VoiceListResponse)
async def list_voices(request: Request) -> VoiceListResponse:
    engine = request.app.state.engine
    voices: list[VoiceInfo] = []

    if CAPABILITIES.builtin_voices and engine is not None:
        for spk in engine.builtin_voices_list:
            if spk == DESIGN_VOICE_ID:
                metadata = {"description": "Voice designed at call time via instructions"}
            else:
                metadata = None
            voices.append(
                VoiceInfo(
                    id=spk, preview_url=None, prompt_text=None, metadata=metadata
                )
            )

    if CAPABILITIES.clone:
        for v in _catalog(request).list():
            voices.append(
                VoiceInfo(
                    id=v.uri,
                    preview_url=f"/v1/audio/voices/preview?id={quote(v.id, safe='')}",
                    prompt_text=v.prompt_text,
                    metadata=v.metadata,
                )
            )

    return VoiceListResponse(voices=voices)


if CAPABILITIES.clone:

    @app.get("/v1/audio/voices/preview")
    async def voice_preview(id: str, request: Request) -> FileResponse:
        voice = _catalog(request).get(id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"voice '{id}' not found")
        return FileResponse(
            voice.wav_path,
            media_type="audio/wav",
            filename=f"{voice.id}.wav",
        )


@app.get("/v1/audio/languages", response_model=LanguagesResponse)
async def list_languages(request: Request) -> LanguagesResponse:
    engine = _engine(request)
    return LanguagesResponse(
        languages=[Language(key=k, name=n) for k, n in engine.list_languages()]
    )


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, request: Request) -> Response:
    settings = _settings(request)
    engine = _engine(request)

    _validate_text(req.input, settings.max_input_chars)
    fmt = _resolve_format(req.response_format, settings)
    kind, voice_obj = _resolve_voice(req.voice, request)
    extras = _speech_extras(req)

    async with _limiter(request).acquire():
        try:
            if kind == "clone":
                assert voice_obj is not None
                samples = await engine.synthesize_clone(
                    req.input,
                    ref_audio=str(voice_obj.wav_path),
                    ref_text=voice_obj.prompt_text,
                    ref_mtime=voice_obj.mtime,
                    language=req.language,
                    **extras,
                )
            else:
                samples = await engine.synthesize_builtin(
                    req.input,
                    voice=req.voice,
                    instructions=req.instructions,
                    language=req.language,
                    **extras,
                )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            log.exception("inference failed")
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}")

        try:
            body, ctype = encode(samples, engine.sample_rate, fmt)
        except Exception as exc:
            log.exception("encoding failed")
            raise HTTPException(status_code=500, detail=f"encoding failed: {exc}")

    return Response(content=body, media_type=ctype)


if CAPABILITIES.clone:

    @app.post("/v1/audio/clone")
    async def clone(
        request: Request,
        audio: UploadFile = File(...),
        prompt_text: str = Form(...),
        input: str = Form(...),
        response_format: Optional[str] = Form(None),
        speed: float = Form(1.0),
        instructions: Optional[str] = Form(None),  # noqa: ARG001 — accepted for compat
        model: Optional[str] = Form(None),  # noqa: ARG001 — accepted for compat
        language: Optional[str] = Form(None),
    ) -> Response:
        settings = _settings(request)
        engine = _engine(request)

        if not 0.25 <= speed <= 4.0:
            raise HTTPException(status_code=422, detail="speed must be in [0.25, 4.0]")
        if not prompt_text.strip():
            raise HTTPException(status_code=422, detail="prompt_text must not be empty")
        _validate_text(input, settings.max_input_chars)
        fmt = _resolve_format(response_format, settings)

        suffix = Path(audio.filename or "").suffix.lower() or ".wav"
        if suffix not in CLONE_AUDIO_EXTS:
            raise HTTPException(
                status_code=415, detail=f"audio format not supported: {suffix}"
            )

        tmp_dir = Path(tempfile.gettempdir()) / "qwen3-tts-open-tts"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

        size = 0
        try:
            with tmp.open("wb") as dest:
                while True:
                    chunk = await audio.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > settings.max_audio_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"audio exceeds {settings.max_audio_bytes} bytes",
                        )
                    dest.write(chunk)

            if size == 0:
                raise HTTPException(status_code=400, detail="audio file is empty")

            async with _limiter(request).acquire():
                try:
                    samples = await engine.synthesize_clone(
                        input,
                        ref_audio=str(tmp),
                        ref_text=prompt_text,
                        ref_mtime=None,
                        language=language,
                    )
                except HTTPException:
                    raise
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc))
                except Exception as exc:
                    log.exception("clone inference failed")
                    raise HTTPException(
                        status_code=500, detail=f"inference failed: {exc}"
                    )

                try:
                    body, ctype = encode(samples, engine.sample_rate, fmt)
                except Exception as exc:
                    log.exception("clone encoding failed")
                    raise HTTPException(
                        status_code=500, detail=f"encoding failed: {exc}"
                    )

            return Response(content=body, media_type=ctype)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                log.warning("failed to unlink temp file %s", tmp)


if CAPABILITIES.design:

    @app.post("/v1/audio/design")
    async def design(req: DesignRequest, request: Request) -> Response:
        settings = _settings(request)
        engine = _engine(request)

        _validate_text(req.input, settings.max_input_chars)
        fmt = _resolve_format(req.response_format, settings)

        extras = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
            "max_new_tokens": req.max_new_tokens,
        }

        async with _limiter(request).acquire():
            try:
                samples = await engine.synthesize_design(
                    req.input,
                    instruct=req.instruct,
                    language=req.language,
                    **extras,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            except Exception as exc:
                log.exception("design inference failed")
                raise HTTPException(status_code=500, detail=f"inference failed: {exc}")

            try:
                body, ctype = encode(samples, engine.sample_rate, fmt)
            except Exception as exc:
                log.exception("design encoding failed")
                raise HTTPException(status_code=500, detail=f"encoding failed: {exc}")

        return Response(content=body, media_type=ctype)
