#!/usr/bin/env python3
"""WebSocket server that receives streaming audio, runs ONNX VAD, and saves speech segments."""

import asyncio
import json
import logging
import os
import struct
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
import websockets
import kaldiio
import kaldi_native_fbank as knf
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SHIFT_MS = 10
FRAME_SHIFT_SAMPLES = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)  # 160
FRAME_PER_SECOND = int(1000 / FRAME_SHIFT_MS)  # 100

AED_LABELS = ["speech", "music", "noise"]


class OnnxStreamVad:
    """Streaming VAD using the ONNX model with caches."""

    def __init__(self, model_path, cmvn_path, config=None):
        self.sess = ort.InferenceSession(model_path)
        self.fbank = FbankExtractor()
        self.cmvn = load_cmvn(cmvn_path)

        # Cache dimensions from model
        # Stream-VAD: 8 blocks, P=128, lookback_padding=19
        self.num_blocks = 8
        self.P = 128
        self.lookback_padding = 19
        self.reset_caches()

        # Config
        self.smooth_window_size = config.get("smooth_window_size", 5) if config else 5
        self.speech_threshold = config.get("speech_threshold", 0.4) if config else 0.4
        self.pad_start_frame = config.get("pad_start_frame", 5) if config else 5
        self.min_speech_frame = config.get("min_speech_frame", 8) if config else 8
        self.max_speech_frame = config.get("max_speech_frame", 2000) if config else 2000
        self.min_silence_frame = config.get("min_silence_frame", 20) if config else 20

        self.pad_start_frame = max(self.smooth_window_size, self.pad_start_frame)
        self.reset_state()

    def reset_caches(self):
        self.caches = np.zeros(
            (self.num_blocks, 1, self.P, self.lookback_padding), dtype=np.float32
        )

    def reset_state(self):
        self.frame_cnt = 0
        self.smooth_window = deque()
        self.smooth_window_sum = 0.0
        self.state = "SILENCE"
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max_speech = False
        self.last_speech_start_frame = -1
        self.last_speech_end_frame = -1

    def reset(self):
        self.reset_caches()
        self.reset_state()

    def process_audio_chunk(self, pcm_int16):
        """Process a chunk of int16 PCM audio. Returns list of (event, start_frame, end_frame)."""
        fbank = self.fbank.extract(pcm_int16)
        if fbank is None or len(fbank) == 0:
            return []

        # Apply CMVN
        fbank = (fbank - self.cmvn["means"]) * self.cmvn["inv_std"]

        # Run ONNX inference on the chunk
        feat = fbank.astype(np.float32)[np.newaxis, :, :]  # (1, T, 80)
        probs, self.caches = self.sess.run(
            None, {"feat": feat, "caches_in": self.caches}
        )
        probs = probs.squeeze()  # type: ignore[union-attr]  # (T,) or scalar
        if probs.ndim == 0:
            probs = [float(probs)]
        else:
            probs = probs.tolist()

        # Process each frame through the state machine
        events = []
        for raw_prob in probs:
            event = self._process_frame(raw_prob)
            if event:
                events.append(event)
        return events

    def _process_frame(self, raw_prob):
        self.frame_cnt += 1

        # Smooth
        self.smooth_window.append(raw_prob)
        self.smooth_window_sum += raw_prob
        if len(self.smooth_window) > self.smooth_window_size:
            self.smooth_window_sum -= self.smooth_window.popleft()
        smoothed = self.smooth_window_sum / len(self.smooth_window)

        is_speech = smoothed >= self.speech_threshold

        event = None

        if self.hit_max_speech:
            event = ("speech_start", self.frame_cnt, self.frame_cnt)
            self.last_speech_start_frame = self.frame_cnt
            self.hit_max_speech = False

        if self.state == "SILENCE":
            if is_speech:
                self.state = "POSSIBLE_SPEECH"
                self.speech_cnt = 1
            else:
                self.silence_cnt += 1
                self.speech_cnt = 0

        elif self.state == "POSSIBLE_SPEECH":
            if is_speech:
                self.speech_cnt += 1
                if self.speech_cnt >= self.min_speech_frame:
                    self.state = "SPEECH"
                    start = max(
                        1,
                        self.frame_cnt - self.speech_cnt + 1 - self.pad_start_frame,
                        self.last_speech_end_frame + 1,
                    )
                    self.last_speech_start_frame = start
                    self.silence_cnt = 0
                    event = ("speech_start", start, self.frame_cnt)
            else:
                self.state = "SILENCE"
                self.silence_cnt = 1
                self.speech_cnt = 0

        elif self.state == "SPEECH":
            self.speech_cnt += 1
            if is_speech:
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    event = (
                        "speech_end",
                        self.last_speech_start_frame,
                        self.frame_cnt,
                    )
                    self.last_speech_end_frame = self.frame_cnt
                    self.last_speech_start_frame = -1
            else:
                self.state = "POSSIBLE_SILENCE"
                self.silence_cnt = 1

        elif self.state == "POSSIBLE_SILENCE":
            self.speech_cnt += 1
            if is_speech:
                self.state = "SPEECH"
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    event = (
                        "speech_end",
                        self.last_speech_start_frame,
                        self.frame_cnt,
                    )
                    self.last_speech_end_frame = self.frame_cnt
                    self.last_speech_start_frame = -1
            else:
                self.silence_cnt += 1
                if self.silence_cnt >= self.min_silence_frame:
                    self.state = "SILENCE"
                    event = (
                        "speech_end",
                        self.last_speech_start_frame,
                        self.frame_cnt,
                    )
                    self.last_speech_end_frame = self.frame_cnt
                    self.last_speech_start_frame = -1
                    self.speech_cnt = 0

        return event

    def flush(self):
        """Call at end of stream to emit any remaining speech segment."""
        if self.state in ("SPEECH", "POSSIBLE_SILENCE"):
            event = (
                "speech_end",
                self.last_speech_start_frame,
                self.frame_cnt,
            )
            self.state = "SILENCE"
            self.last_speech_end_frame = self.frame_cnt
            self.last_speech_start_frame = -1
            self.speech_cnt = 0
            return event
        return None


class FbankExtractor:
    """Kaldi-compatible 80-dim fbank feature extraction."""

    def __init__(self):
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = FRAME_SHIFT_MS
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        opts.mel_opts.debug_mel = False
        self.opts = opts
        self.remainder = np.array([], dtype=np.int16)

    def extract(self, pcm_int16):
        """Extract fbank features from int16 PCM samples. Handles partial frames."""
        # Concatenate with remainder from previous chunk
        samples = np.concatenate([self.remainder, pcm_int16])

        fbank = knf.OnlineFbank(self.opts)
        fbank.accept_waveform(SAMPLE_RATE, samples.tolist())

        num_frames = fbank.num_frames_ready
        if num_frames == 0:
            self.remainder = samples
            return None

        feat = []
        for i in range(num_frames):
            feat.append(fbank.get_frame(i))
        feat = np.vstack(feat)

        # Keep unprocessed samples as remainder
        consumed_samples = num_frames * FRAME_SHIFT_SAMPLES
        self.remainder = samples[consumed_samples:]

        return feat

    def reset(self):
        self.remainder = np.array([], dtype=np.int16)


def load_cmvn(cmvn_path):
    """Load Kaldi CMVN stats."""
    import math

    stats: np.ndarray = kaldiio.load_mat(cmvn_path)  # type: ignore[assignment]
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    means = stats[0, :dim] / count
    variances = (stats[1, :dim] / count) - means**2
    variances = np.maximum(variances, 1e-20)
    inv_std = 1.0 / np.sqrt(variances)
    return {"means": means.astype(np.float32), "inv_std": inv_std.astype(np.float32)}


class OnnxAed:
    """Audio Event Detection using the AED ONNX model."""

    def __init__(self, model_path, cmvn):
        self.sess = ort.InferenceSession(model_path)
        self.cmvn = cmvn

    def classify_segment(self, pcm_int16):
        """Classify an audio segment. Returns (label, probabilities dict)."""
        fbank = self._extract_fbank(pcm_int16)
        if fbank is None or len(fbank) == 0:
            return None, {}

        fbank = (fbank - self.cmvn["means"]) * self.cmvn["inv_std"]
        feat = fbank.astype(np.float32)[np.newaxis, :, :]  # (1, T, 80)

        outputs = self.sess.run(None, {"feat": feat})
        probs = outputs[0].squeeze()  # type: ignore[union-attr]  # (T, 3)

        # Average probabilities across all frames
        if probs.ndim == 1:
            avg_probs = probs
        else:
            avg_probs = probs.mean(axis=0)

        label_idx = int(np.argmax(avg_probs))
        label = AED_LABELS[label_idx]
        prob_dict = {AED_LABELS[i]: round(float(avg_probs[i]), 4) for i in range(len(AED_LABELS))}
        return label, prob_dict

    def _extract_fbank(self, pcm_int16):
        """Extract fbank features from a complete audio segment."""
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = FRAME_SHIFT_MS
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        opts.mel_opts.debug_mel = False

        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(SAMPLE_RATE, pcm_int16.tolist())

        num_frames = fbank.num_frames_ready
        if num_frames == 0:
            return None

        feat = []
        for i in range(num_frames):
            feat.append(fbank.get_frame(i))
        return np.vstack(feat)


class SessionState:
    """Per-connection state."""

    def __init__(self, vad, output_dir):
        self.vad = vad
        self.output_dir = output_dir
        self.audio_buffer = []  # all received int16 samples
        self.total_samples = 0
        self.segment_count = 0

    def add_audio(self, pcm_int16):
        self.audio_buffer.append(pcm_int16)
        self.total_samples += len(pcm_int16)

    def get_segment_audio(self, start_frame, end_frame):
        """Extract audio samples for a given frame range."""
        start_sample = max(0, (start_frame - 1) * FRAME_SHIFT_SAMPLES)
        end_sample = min(self.total_samples, end_frame * FRAME_SHIFT_SAMPLES)
        all_audio = np.concatenate(self.audio_buffer)
        segment = all_audio[start_sample:end_sample]
        return segment if len(segment) > 0 else None

    def save_segment(self, start_frame, end_frame):
        """Save a speech segment to a WAV file."""
        start_sample = max(0, (start_frame - 1) * FRAME_SHIFT_SAMPLES)
        end_sample = min(self.total_samples, end_frame * FRAME_SHIFT_SAMPLES)

        all_audio = np.concatenate(self.audio_buffer)
        segment = all_audio[start_sample:end_sample]

        if len(segment) == 0:
            return None

        self.segment_count += 1
        start_s = start_sample / SAMPLE_RATE
        end_s = end_sample / SAMPLE_RATE
        filename = f"segment_{self.segment_count:04d}_{start_s:.2f}s_{end_s:.2f}s.wav"
        filepath = os.path.join(self.output_dir, filename)

        sf.write(filepath, segment, SAMPLE_RATE, subtype="PCM_16")
        dur = len(segment) / SAMPLE_RATE
        logger.info(f"Saved {filepath} ({dur:.2f}s)")
        return filepath


async def handle_client(websocket, model_path, cmvn_path, aed_model_path, output_dir):
    """Handle a single WebSocket connection."""
    client_id = id(websocket)
    session_dir = os.path.join(output_dir, f"session_{client_id}_{int(time.time())}")
    os.makedirs(session_dir, exist_ok=True)
    logger.info(f"New connection {client_id}, saving to {session_dir}")

    vad = OnnxStreamVad(model_path, cmvn_path)
    aed = OnnxAed(aed_model_path, vad.cmvn) if aed_model_path else None
    session = SessionState(vad, session_dir)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Raw int16 LE PCM audio
                pcm_int16 = np.frombuffer(message, dtype=np.int16)
                session.add_audio(pcm_int16)

                events = vad.process_audio_chunk(pcm_int16)

                for event_type, start_frame, end_frame in events:
                    start_s = (start_frame - 1) / FRAME_PER_SECOND
                    end_s = end_frame / FRAME_PER_SECOND

                    if event_type == "speech_start":
                        logger.info(f"[{client_id}] Speech started at {start_s:.2f}s")
                        await websocket.send(
                            json.dumps({"event": "speech_start", "time": round(start_s, 3)})
                        )

                    elif event_type == "speech_end":
                        logger.info(f"[{client_id}] Speech ended at {end_s:.2f}s")
                        filepath = session.save_segment(start_frame, end_frame)
                        msg = {
                            "event": "speech_end",
                            "start": round(start_s, 3),
                            "end": round(end_s, 3),
                            "file": filepath,
                        }
                        if aed:
                            segment_audio = session.get_segment_audio(start_frame, end_frame)
                            if segment_audio is not None:
                                label, probs = aed.classify_segment(segment_audio)
                                msg["aed_label"] = label
                                msg["aed_probs"] = probs
                                logger.info(f"[{client_id}] AED: {label} {probs}")
                        await websocket.send(json.dumps(msg))

            elif isinstance(message, str):
                msg = json.loads(message)
                if msg.get("action") == "reset":
                    vad.reset()
                    session.audio_buffer = []
                    session.total_samples = 0
                    logger.info(f"[{client_id}] Reset")

        # Client disconnected — flush any remaining speech
        event = vad.flush()
        if event:
            event_type, start_frame, end_frame = event
            filepath = session.save_segment(start_frame, end_frame)
            if filepath:
                logger.info(f"[{client_id}] Flushed final segment: {filepath}")

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection {client_id} closed")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="FireRedVAD WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model", default="onnx_models/fireredvad_stream_vad_with_cache.onnx"
    )
    parser.add_argument("--cmvn", default="onnx_models/cmvn.ark")
    parser.add_argument(
        "--aed-model", default="onnx_models/fireredvad_aed.onnx"
    )
    parser.add_argument("--output-dir", default="vad_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Verify model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        logger.error("Run: python export_onnx_streaming.py")
        return
    if not os.path.exists(args.cmvn):
        logger.error(f"CMVN not found: {args.cmvn}")
        return

    aed_model_path = args.aed_model if os.path.exists(args.aed_model) else None
    if aed_model_path:
        logger.info(f"Loading AED model: {aed_model_path}")
    else:
        logger.info("AED model not found, running without audio event detection")

    logger.info(f"Loading model: {args.model}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Listening on ws://{args.host}:{args.port}")

    async with websockets.serve(
        lambda ws: handle_client(ws, args.model, args.cmvn, aed_model_path, args.output_dir),
        args.host,
        args.port,
        max_size=2**20,  # 1MB max message
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
