#!/usr/bin/env python3
"""Real-time Cantonese text-to-speech over the ElevenLabs WebSocket API.

Uses Eleven v4 Turbo (~100 ms median latency, supports Cantonese `yue`).
v4 models are only served by the Text-to-Dialogue WebSocket
(/v1/text-to-dialogue/stream-input), not the classic TTS stream-input endpoint.

One WebSocket stays open; every text you send is flushed immediately and the raw
PCM audio is piped to a player as it streams back.

Usage:
    export ELEVENLABS_API_KEY=...
    python realtime_tts.py "你好，歡迎使用即時語音合成。"   # one-shot
    python realtime_tts.py                                   # interactive: one line = one utterance
    python realtime_tts.py "你好" -o out.wav                 # also save a WAV
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import sys
import time
import wave
from urllib.parse import urlencode

import websockets

HOST = "wss://api.elevenlabs.io"
DEFAULT_MODEL = "eleven_v4_turbo"
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # "Rachel"; replace with a Cantonese voice from your library
KEEP_ALIVE_SECONDS = 10  # server drops the session after 20 s of silence


async def open_player(sample_rate: int) -> asyncio.subprocess.Process:
    """Start a low-buffer player that reads raw s16le mono PCM from stdin."""
    if shutil.which("ffplay"):
        cmd = [
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0",
            "-f", "s16le", "-ar", str(sample_rate), "-ch_layout", "mono", "-i", "-",
        ]
    elif shutil.which("play"):  # sox
        cmd = ["play", "-q", "-t", "raw", "-r", str(sample_rate), "-e", "signed",
               "-b", "16", "-c", "1", "-"]
    else:
        sys.exit("No streaming player found. Install ffmpeg (ffplay) or sox, or use --no-play.")
    return await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE)


class Session:
    def __init__(self, args: argparse.Namespace, ws, player, pcm: bytearray | None):
        self.args = args
        self.ws = ws
        self.player = player
        self.pcm = pcm
        self.sent_at: float | None = None  # set on send, cleared at first audio chunk

    async def send_text(self, text: str, close: bool = False) -> None:
        msg = {"inputs": [{"text": text + " ", "voice_id": self.args.voice, "new_turn": True}]}
        if close:
            msg["close_socket"] = True  # flushes buffered text, then ends the session
        else:
            msg["flush"] = True  # generate now instead of waiting for the buffer threshold
        self.sent_at = time.perf_counter()
        await self.ws.send(json.dumps(msg))

    async def receive(self) -> None:
        """Play audio chunks as they arrive until the server signals the end."""
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg.get("error"):
                raise RuntimeError(f"{msg['error']}: {msg.get('message')}")
            if msg.get("audio"):
                chunk = base64.b64decode(msg["audio"])
                if self.sent_at is not None:
                    ms = (time.perf_counter() - self.sent_at) * 1000
                    print(f"[time to first audio: {ms:.0f} ms]", file=sys.stderr)
                    self.sent_at = None
                if self.pcm is not None:
                    self.pcm.extend(chunk)
                if self.player:
                    try:
                        self.player.stdin.write(chunk)
                        await self.player.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        self.player = None
            if msg.get("is_final"):
                return

    async def keep_alive(self) -> None:
        while True:
            await asyncio.sleep(KEEP_ALIVE_SECONDS)
            await self.ws.send(json.dumps({"keep_alive": True}))

    async def read_lines(self) -> None:
        loop = asyncio.get_running_loop()
        interactive = sys.stdin.isatty()
        while True:
            if interactive:
                print("> ", end="", file=sys.stderr, flush=True)
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:  # EOF / Ctrl-D
                break
            if line.strip():
                await self.send_text(line.strip())
        await self.ws.send(json.dumps({"close_socket": True}))


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit("Set ELEVENLABS_API_KEY first.")

    query = urlencode({
        "model_id": args.model,
        "language_code": args.language,
        "output_format": f"pcm_{args.sample_rate}",  # raw PCM: no decoder delay
    })
    url = f"{HOST}/v1/text-to-dialogue/stream-input?{query}"

    player = None if args.no_play else await open_player(args.sample_rate)
    pcm = bytearray() if args.output else None

    t0 = time.perf_counter()
    async with websockets.connect(url, additional_headers={"xi-api-key": api_key}) as ws:
        print(f"[connected in {(time.perf_counter() - t0) * 1000:.0f} ms]", file=sys.stderr)
        await ws.send(json.dumps({"voices": [args.voice]}))

        session = Session(args, ws, player, pcm)
        receiver = asyncio.create_task(session.receive())
        keeper = asyncio.create_task(session.keep_alive())
        try:
            if args.text:
                await session.send_text(args.text, close=True)
            else:
                print("Type Cantonese text, one utterance per line (Ctrl-D to quit).", file=sys.stderr)
                await session.read_lines()
            await receiver
        finally:
            keeper.cancel()
            receiver.cancel()
            player = session.player

    if player:
        try:
            player.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        await player.wait()

    if pcm is not None:
        with wave.open(args.output, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(args.sample_rate)
            w.writeframes(bytes(pcm))
        print(f"saved {args.output}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="text to speak; omit for interactive mode")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"v3/v4 model (default: {DEFAULT_MODEL}; eleven_v4 = highest quality, slower)")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="ElevenLabs voice ID")
    p.add_argument("--language", default="yue", help="language_code (default: yue = Cantonese)")
    p.add_argument("--sample-rate", type=int, default=24000, choices=[16000, 22050, 24000, 44100],
                   help="PCM sample rate (default: 24000)")
    p.add_argument("-o", "--output", help="also save audio to this WAV file")
    p.add_argument("--no-play", action="store_true", help="don't play audio")
    try:
        asyncio.run(run(p.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
