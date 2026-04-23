# qwen3-tts-open-tts

[English](./README.md) · **中文**

基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的 OpenAI 兼容 HTTP TTS
服务。以单个 CUDA 镜像发布到 GHCR。

实现 [Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec)：

- `POST /v1/audio/speech` — OpenAI 兼容的语音合成
- `POST /v1/audio/clone` — 一次性零样本克隆（multipart 上传，仅 `base` 变体）
- `POST /v1/audio/design` — 自然语言声音设计（仅 `voicedesign` 变体）
- `GET  /v1/audio/voices` — 列出文件声音与内置声音
- `GET  /v1/audio/voices/preview?id=...` — 下载参考 WAV（仅 `base` 变体）
- `GET  /v1/audio/languages` — 支持的语言列表
- `GET  /healthz` — 引擎状态、能力矩阵、并发快照

支持 6 种输出格式（`mp3`/`opus`/`aac`/`flac`/`wav`/`pcm`），单声道
`float32`，采样率 24 kHz。声音以
`${VOICES_DIR}/<id>.{wav,txt,yml}` 三元组（或二元组）形式存放在磁盘
（`base` 变体）。

## 变体

Qwen3-TTS 提供三个互斥的模型权重。通过 `QWEN3_VARIANT` 在容器启动时
选择，capabilities 会随之切换：

| 变体 | 默认模型 | `clone` | `builtin_voices` | `design` | `/v1/audio/speech` 的 voice |
|---|---|---|---|---|---|
| `base`（默认） | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ✅ | ❌ | ❌ | `file://<id>` |
| `customvoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | ❌ | ✅（9 个预置说话人） | ❌ | `<speaker>` |
| `voicedesign` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ❌ | ✅（仅 `design` 占位） | ✅ | `design` + `instructions` |

在 `voicedesign` 模式下，唯一的声音 `design` 会把 `/v1/audio/speech`
路由到 `generate_voice_design()`，使用 OpenAI 兼容字段
`instructions` 作为声音描述。

## 快速开始

```bash
mkdir -p voices cache

# base 变体：放入 5-15 秒的参考 WAV 和对应转录文本
cp ~/my-ref.wav voices/alice.wav
echo "这是参考音频的转录文本。" > voices/alice.txt

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/qwen3-tts-open-tts:latest
```

首次启动会下载模型权重（视变体 3-7 GB）到 `/root/.cache`。挂载 cache
目录可避免重复下载。加载期间 `/healthz` 返回 `status="loading"`。

```bash
curl -s localhost:8000/healthz | jq
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，这是 Qwen3-TTS。","voice":"file://alice","response_format":"mp3"}' \
  -o out.mp3
```

## 环境变量

### 引擎相关（前缀 `QWEN3_`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWEN3_VARIANT` | `base` | `base` / `customvoice` / `voicedesign` |
| `QWEN3_MODEL` | *（按变体默认）* | HF 仓库 id 或本地路径；覆盖变体默认值 |
| `QWEN3_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `QWEN3_CUDA_INDEX` | `0` | 多卡时指定 GPU 索引 |
| `QWEN3_DTYPE` | `bfloat16` | `float16` / `bfloat16` / `float32` |
| `QWEN3_ATTN_IMPL` | `flash_attention_2` | `flash_attention_2` / `sdpa` / `eager` |
| `QWEN3_PROMPT_CACHE_SIZE` | `16` | 每个声音 prompt items 的 LRU 容量（仅 `base` 变体） |

### 服务相关（无前缀）

| 变量 | 默认 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | uvicorn 日志级别 |
| `VOICES_DIR` | `/voices` | 文件声音扫描根目录 |
| `MAX_INPUT_CHARS` | `8000` | 超过返回 413 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `MAX_CONCURRENCY` | `1` | 并发合成上限 |
| `MAX_QUEUE_SIZE` | `0` | 0 = 队列无界 |
| `QUEUE_TIMEOUT` | `0` | 秒；0 = 无超时 |
| `MAX_AUDIO_BYTES` | `20971520` | `/v1/audio/clone` 上传大小限制 |
| `CORS_ENABLED` | `false` | 启用面向浏览器客户端的宽松 `CORSMiddleware` |

## Compose

见 [`docker/docker-compose.example.yml`](docker/docker-compose.example.yml)。

## API 请求参数

GET 端点（`/healthz`、`/v1/audio/voices`、`/v1/audio/voices/preview`、
`/v1/audio/languages`）不接收 body，响应结构详见
[Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md)。

下表描述接收 body 的 POST 端点。**状态**列使用固定词汇：

- **required** — 缺失返回 422
- **supported** — 被 Qwen3-TTS 识别并使用
- **ignored** — 为 OpenAI 兼容接收但无效果
- **conditional** — 行为取决于其他字段或变体
- **extension** — Qwen3-TTS 私有字段，非规范定义

### `POST /v1/audio/speech` (application/json)

| 字段 | 类型 | 默认 | 状态 | 说明 |
|---|---|---|---|---|
| `model` | string | `null` | ignored | 仅用于 OpenAI 兼容 |
| `input` | string | — | required | 1..`MAX_INPUT_CHARS` 字符；空返回 422，超长返回 413 |
| `voice` | string | — | required | `file://<id>`（base）、`<speaker>`（customvoice）或 `design`（voicedesign） |
| `response_format` | enum | `mp3` | supported | `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm` |
| `speed` | float | `1.0` | ignored | 仅用于 OpenAI 兼容；Qwen3-TTS 不暴露 speed |
| `instructions` | string \| null | `null` | conditional | `customvoice`/`voicedesign` 使用；`base` 忽略 |
| `language` | string \| null | `Auto` | extension | 取自 `/v1/audio/languages.key` |
| `temperature` | float \| null | `null` | extension | 范围 `[0.0, 2.0]`，转发给 `generate()` |
| `top_p` | float \| null | `null` | extension | 范围 `[0.0, 1.0]` |
| `top_k` | int \| null | `null` | extension | `0` 关闭 |
| `repetition_penalty` | float \| null | `null` | extension | 范围 `[0.5, 2.0]` |
| `max_new_tokens` | int \| null | `null` | extension | 生成 codec token 上限 |

### `POST /v1/audio/clone` (multipart/form-data，仅 `base` 变体)

| 字段 | 类型 | 默认 | 状态 | 说明 |
|---|---|---|---|---|
| `audio` | file | — | required | 扩展名须为 `.wav/.mp3/.flac/.ogg/.opus/.m4a/.aac/.webm`；超过 `MAX_AUDIO_BYTES` 返回 413 |
| `prompt_text` | string | — | required | 参考音频的转录文本 |
| `input` | string | — | required | 同 `/speech.input` |
| `response_format` | string | `mp3` | supported | 同 `/speech` |
| `speed` | float | `1.0` | ignored | 同 `/speech.speed` |
| `instructions` | string \| null | `null` | ignored | `base` 变体不支持风格提示 |
| `model` | string | `null` | ignored | 仅用于 OpenAI 兼容 |
| `language` | string \| null | `Auto` | extension | 同 `/speech.language` |

### `POST /v1/audio/design` (application/json，仅 `voicedesign` 变体)

| 字段 | 类型 | 默认 | 状态 | 说明 |
|---|---|---|---|---|
| `input` | string | — | required | 1..`MAX_INPUT_CHARS` 字符 |
| `instruct` | string \| null | `null` | supported | 自然语言声音描述；`null`/`""` 使用引擎默认声音 |
| `response_format` | string | `mp3` | supported | 同 `/speech` |
| `language` | string \| null | `Auto` | extension | 同 `/speech.language` |
| `temperature`/`top_p`/`top_k`/`repetition_penalty`/`max_new_tokens` | — | — | extension | 同 `/speech` |

## 已知限制

- 不暴露流式端点（`/v1/audio/realtime`）。上游 wrapper 的
  `non_streaming_mode=False` 只模拟流式文本输入，并不产生增量音频块。
- `speed` 字段仅为 OpenAI 兼容接收，运行时无实际效果。
- `base` 变体下首次请求某个声音会付出说话人嵌入提取成本；相同
  `(path, mtime)` 的后续请求命中 prompt LRU。`/v1/audio/clone` 的 multipart
  路径始终绕过 LRU。
- 默认预装并启用 `flash-attn`；若硬件不支持，可设
  `QWEN3_ATTN_IMPL=sdpa` 退回 PyTorch 内建 SDPA。
