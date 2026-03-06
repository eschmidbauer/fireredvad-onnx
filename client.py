#!/usr/bin/env python3
"""WebSocket client that streams audio to the VAD server.

Supports two modes:
  1. Stream from a WAV file:   python client.py --file audio.wav
  2. Stream from microphone:   python client.py --mic
"""

import asyncio
import argparse
import json
import logging
import sys

import numpy as np
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Send audio in 100ms chunks (1600 samples = 3200 bytes)
CHUNK_SAMPLES = 1600
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE


async def stream_file(uri, filepath):
    """Stream a WAV file to the server."""
    import soundfile as sf

    wav, sr = sf.read(filepath, dtype="int16")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz. "
                         f"Convert with: ffmpeg -i {filepath} -ar 16000 -ac 1 -acodec pcm_s16le out.wav")
    if wav.ndim > 1:
        wav = wav[:, 0]  # take first channel

    total_dur = len(wav) / SAMPLE_RATE
    logger.info(f"Streaming {filepath} ({total_dur:.1f}s) to {uri}")

    async with websockets.connect(uri) as ws:
        recv_task = asyncio.create_task(receive_events(ws))

        # Stream in chunks, simulating real-time
        offset = 0
        while offset < len(wav):
            chunk = wav[offset : offset + CHUNK_SAMPLES]
            await ws.send(chunk.tobytes())
            offset += CHUNK_SAMPLES
            # Simulate real-time playback speed
            await asyncio.sleep(CHUNK_DURATION)

        # Small delay to let server process final frames
        await asyncio.sleep(0.5)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    logger.info("Done streaming file")


async def stream_mic(uri):
    """Stream from microphone to the server."""
    try:
        import pyaudio
    except ImportError:
        logger.error("pyaudio is required for mic mode: pip install pyaudio")
        sys.exit(1)

    pa = pyaudio.PyAudio()

    logger.info(f"Streaming microphone to {uri}")
    logger.info("Press Ctrl+C to stop")

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )

    async with websockets.connect(uri) as ws:
        recv_task = asyncio.create_task(receive_events(ws))

        try:
            while True:
                data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                await ws.send(data)
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass


async def receive_events(ws):
    """Listen for events from the server and print them."""
    try:
        async for message in ws:
            event = json.loads(message)
            if event["event"] == "speech_start":
                logger.info(f">> Speech START at {event['time']:.3f}s")
            elif event["event"] == "speech_end":
                dur = event["end"] - event["start"]
                logger.info(
                    f">> Speech END   at {event['end']:.3f}s "
                    f"(duration: {dur:.3f}s, saved: {event.get('file', 'N/A')})"
                )
    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        pass


def main():
    parser = argparse.ArgumentParser(description="FireRedVAD WebSocket Client")
    parser.add_argument("--uri", default="ws://localhost:8765", help="WebSocket URI")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="WAV file to stream (16kHz 16-bit mono)")
    group.add_argument("--mic", action="store_true", help="Stream from microphone")

    args = parser.parse_args()

    if args.file:
        asyncio.run(stream_file(args.uri, args.file))
    else:
        asyncio.run(stream_mic(args.uri))


if __name__ == "__main__":
    main()
