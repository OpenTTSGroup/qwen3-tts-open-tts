# qwen3-tts-open-tts

**English** · [中文](./README.zh.md)

OpenAI-compatible HTTP TTS service built on top of
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Ships as a single CUDA
container image on GHCR.

Implements the [Open TTS spec](https://github.com/OpenTTSGroup/open-tts-spec):

- `POST /v1/audio/speech` — OpenAI-compatible synthesis
- `POST /v1/audio/clone` — one-shot zero-shot cloning (multipart upload, `base` variant only)
- `POST /v1/audio/design` — natural-language voice design (`voicedesign` variant only)
- `GET  /v1/audio/voices` — list file-based and built-in voices
- `GET  /v1/audio/voices/preview?id=...` — download a reference WAV (`base` variant only)
- `GET  /v1/audio/languages` — supported language keys
- `GET  /healthz` — engine status, capabilities, concurrency snapshot

Six output formats (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`); mono
`float32` at 24 kHz. Voices live on disk as
`${VOICES_DIR}/<id>.{wav,txt,yml}` triples (`base` variant).

## Variants

Qwen3-TTS ships three mutually-exclusive model checkpoints. Pick one at
container start time via `QWEN3_VARIANT`; capabilities toggle accordingly:

| variant | default model | `clone` | `builtin_voices` | `design` | `/v1/audio/speech` voice |
|---|---|---|---|---|---|
| `base` *(default)* | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ✅ | ❌ | ❌ | `file://<id>` |
| `customvoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | ❌ | ✅ (9 named speakers) | ❌ | `<speaker>` |
| `voicedesign` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ❌ | ✅ (single `design` voice) | ✅ | `design` + `instructions` |

In `voicedesign` mode the sole "voice" `design` routes `/v1/audio/speech`
through `generate_voice_design()`, using the OpenAI-compatible
`instructions` field as the voice description.

## Quick start

```bash
mkdir -p voices cache

# base variant: drop a 5–15 s reference WAV plus its transcript.
cp ~/my-ref.wav voices/alice.wav
echo "This is the transcript of the reference clip." > voices/alice.txt

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/qwen3-tts-open-tts:latest
```

First boot downloads the model weights (~3-7 GB depending on variant) to
`/root/.cache`. Mount the cache directory to avoid repeat downloads.
`/healthz` reports `status="loading"` until the engine is ready.

```bash
curl -s localhost:8000/healthz | jq
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Qwen3-TTS.","voice":"file://alice","response_format":"mp3"}' \
  -o out.mp3
```

## Environment variables

### Engine (prefixed `QWEN3_`)

| variable | default | description |
|---|---|---|
| `QWEN3_VARIANT` | `base` | `base` / `customvoice` / `voicedesign` |
| `QWEN3_MODEL` | *(per-variant)* | HF repo id or local path; overrides the variant default |
| `QWEN3_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `QWEN3_CUDA_INDEX` | `0` | GPU index when multiple are visible |
| `QWEN3_DTYPE` | `bfloat16` | `float16` / `bfloat16` / `float32` |
| `QWEN3_ATTN_IMPL` | `flash_attention_2` | `flash_attention_2` / `sdpa` / `eager` |
| `QWEN3_PROMPT_CACHE_SIZE` | `16` | LRU size for per-voice prompt items (`base` variant only) |

### Service-level (no prefix)

| variable | default | description |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `VOICES_DIR` | `/voices` | scan root for file-based voices |
| `MAX_INPUT_CHARS` | `8000` | 413 above this |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `MAX_CONCURRENCY` | `1` | in-flight synthesis ceiling |
| `MAX_QUEUE_SIZE` | `0` | 0 = unbounded queue |
| `QUEUE_TIMEOUT` | `0` | seconds; 0 = unbounded wait |
| `MAX_AUDIO_BYTES` | `20971520` | upload limit for `/v1/audio/clone` |
| `CORS_ENABLED` | `false` | enable a permissive `CORSMiddleware` for browser clients. |

## Compose

See [`docker/docker-compose.example.yml`](docker/docker-compose.example.yml).

## API request parameters

GET endpoints (`/healthz`, `/v1/audio/voices`, `/v1/audio/voices/preview`,
`/v1/audio/languages`) take no body. See the
[Open TTS spec](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md)
for their response shape.

The tables below describe the POST endpoints that accept a request body.
The **Status** column uses a fixed vocabulary:

- **required** — rejected with 422 if missing.
- **supported** — accepted and consumed by Qwen3-TTS.
- **ignored** — accepted for OpenAI compatibility; has no effect.
- **conditional** — behaviour depends on other fields or the loaded variant.
- **extension** — Qwen3-TTS-specific, not part of the Open TTS spec.

### `POST /v1/audio/speech` (application/json)

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `model` | string | `null` | ignored | OpenAI compatibility only. |
| `input` | string | — | required | 1..`MAX_INPUT_CHARS` chars. Empty ⇒ 422, over limit ⇒ 413. |
| `voice` | string | — | required | `file://<id>` (base), `<speaker>` (customvoice), or `design` (voicedesign). |
| `response_format` | enum | `mp3` | supported | `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm`. |
| `speed` | float | `1.0` | ignored | Accepted for OpenAI compatibility; Qwen3-TTS does not expose a speed knob. |
| `instructions` | string \| null | `null` | conditional | Used by `customvoice` and `voicedesign` as the style prompt. Ignored by `base`. |
| `language` | string \| null | `Auto` | extension | One of `/v1/audio/languages.key`. |
| `temperature` | float \| null | `null` | extension | Forwarded to `generate()`. Range `[0.0, 2.0]`. |
| `top_p` | float \| null | `null` | extension | Range `[0.0, 1.0]`. |
| `top_k` | int \| null | `null` | extension | `0` disables. |
| `repetition_penalty` | float \| null | `null` | extension | Range `[0.5, 2.0]`. |
| `max_new_tokens` | int \| null | `null` | extension | Upper bound on generated codec tokens. |

### `POST /v1/audio/clone` (multipart/form-data, `base` variant only)

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `audio` | file | — | required | Extension one of `.wav/.mp3/.flac/.ogg/.opus/.m4a/.aac/.webm`. 413 above `MAX_AUDIO_BYTES`. |
| `prompt_text` | string | — | required | Reference-clip transcript. |
| `input` | string | — | required | Same as `/speech.input`. |
| `response_format` | string | `mp3` | supported | Same as `/speech`. |
| `speed` | float | `1.0` | ignored | Same as `/speech.speed`. |
| `instructions` | string \| null | `null` | ignored | `base` variant does not support style prompts. |
| `model` | string | `null` | ignored | OpenAI compatibility only. |
| `language` | string \| null | `Auto` | extension | Same as `/speech.language`. |

### `POST /v1/audio/design` (application/json, `voicedesign` variant only)

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `input` | string | — | required | 1..`MAX_INPUT_CHARS` chars. |
| `instruct` | string \| null | `null` | supported | Natural-language voice description; `null`/`""` uses the engine default voice. |
| `response_format` | string | `mp3` | supported | Same as `/speech`. |
| `language` | string \| null | `Auto` | extension | Same as `/speech.language`. |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_new_tokens` | — | — | extension | Same as `/speech`. |

## Known limitations

- Streaming (`/v1/audio/realtime`) is **not** exposed. The upstream
  wrapper's `non_streaming_mode=False` only simulates streaming text input
  and does not yield incremental audio chunks.
- `speed` is accepted for OpenAI compatibility but has no runtime effect.
- First call for a new `voice` on the `base` path pays a speaker-embedding
  cost; subsequent calls for the same `(path, mtime)` hit the prompt LRU.
  The `/v1/audio/clone` multipart path always skips the LRU.
- The `flash-attn` wheel is preinstalled and used by default. Set
  `QWEN3_ATTN_IMPL=sdpa` to fall back to PyTorch's built-in SDPA on
  hardware that cannot run flash-attn.
