# fireredvad-onnx

Streaming Voice Activity Detection over WebSocket using [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) ONNX models. Includes optional Audio Event Detection (AED) to classify speech segments as speech, music, or noise.

## Requirements

- Python 3.10+
- ONNX model files in `onnx_models/`:
  - `fireredvad_stream_vad_with_cache.onnx` — streaming VAD model
  - `cmvn.ark` — CMVN normalization stats
  - `fireredvad_aed.onnx` — audio event detection model (optional)

## Install

```bash
uv sync
```

## Server

The server accepts streaming 16kHz 16-bit mono PCM audio over WebSocket, runs VAD to detect speech segments, and optionally classifies each segment using the AED model.

```bash
python server.py
```

### Server Options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8765` | WebSocket port |
| `--model` | `onnx_models/fireredvad_stream_vad_with_cache.onnx` | VAD model path |
| `--cmvn` | `onnx_models/cmvn.ark` | CMVN stats path |
| `--aed-model` | `onnx_models/fireredvad_aed.onnx` | AED model path (skipped if not found) |
| `--output-dir` | `vad_output` | Directory for saved audio segments |

### WebSocket protocol

**Client sends:**

- Binary messages: raw int16 little-endian PCM audio at 16kHz
- JSON messages: `{"action": "reset"}` to reset VAD state

**Server sends:**

Speech start:

```json
{"event": "speech_start", "time": 1.234}
```

Speech end (with AED when enabled):

```json
{
  "event": "speech_end",
  "start": 1.234,
  "end": 3.456,
  "file": "vad_output/session_.../segment_0001_1.23s_3.46s.wav",
  "aed_label": "speech",
  "aed_probs": {"speech": 0.95, "music": 0.03, "noise": 0.02}
}
```

## Client

The client streams audio to the server and prints VAD events.

### Stream a WAV file

```bash
python client.py --file audio.wav
```

The file must be 16kHz 16-bit mono WAV. Convert with ffmpeg if needed:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -acodec pcm_s16le audio.wav
```

### Stream from microphone

```bash
python client.py --mic
```

Press `Ctrl+C` to stop.

### Client Options

| Flag | Default | Description |
| ---- | ------- | -----------|
| `--uri` | `ws://localhost:8765` | WebSocket server URI |
| `--file` | | WAV file to stream |
| `--mic` | | Stream from microphone |
