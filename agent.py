#!/usr/bin/env python3
"""Cantonese voice agent: talk into the microphone, hear the reply from the speaker.

    mic -> silence detection -> ElevenLabs Scribe          (transcribe.py)
        -> OpenRouter chat model, streamed
        -> sentence chunks -> ElevenLabs realtime TTS      (realtime_tts.py)
        -> speaker

Latency tricks: the TTS WebSocket is opened while the LLM is still thinking, and every
sentence is sent to TTS (and starts playing) as soon as the LLM has produced it.

The mic is ignored while the agent is speaking so it doesn't hear itself (no barge-in);
headphones are not required.

Usage:
    export ELEVENLABS_API_KEY=sk_...
    export OPENROUTER_API_KEY=sk-or-...
    python agent.py --list-devices
    python agent.py --voice <cantonese_voice_id>
    python agent.py --voice <id> --llm deepseek/deepseek-chat --input-device 0
Press Ctrl-C to quit.
"""

import argparse
import asyncio
import base64
import io
import json
import math
import os
import queue
import re
import sys
import threading
import time
import wave
from array import array
from collections import deque
from urllib.parse import urlencode

import requests
import sounddevice as sd
import websockets
from opencc import OpenCC

import realtime_tts
import transcribe

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM = "x-ai/grok-4.3"  # comma-separated list = fallback order

SYSTEM_PROMPT = (
    "你係一個語音助手，同用家用香港廣東話傾偈。\n"
    "- 用地道口語廣東話回答（例如：係、唔、嘅、咗、喺、啲、嘢），用繁體中文，唔好用書面語。\n"
    "- 回答要簡短，一至三句，似真人講嘢。\n"
    "- 答案會用聲音讀出嚟，所以唔好用 markdown、列表、表情符號或者特殊符號。\n"
    "- 用戶嘅說話係語音辨識出嚟，可能有錯字，請按上文理解。"
)

MIC_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = MIC_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
PREROLL_FRAMES = 10   # audio kept from before speech was detected (300 ms)
START_FRAMES = 3      # consecutive loud frames that count as speech starting (90 ms)
MIN_LOUD_FRAMES = 8   # utterances with less speech than this (240 ms) are treated as noise

_S2HK = OpenCC("s2hk")
_MARKDOWN = re.compile(r"[*_`#>~|]")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")


def clean(text: str) -> str:
    """Make LLM output safe to read aloud: no markdown/emoji, Traditional (Hong Kong) script."""
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    text = re.sub(r"^\s*[-•]\s+", "", text)
    return _S2HK.convert(re.sub(r"\s+", " ", text)).strip()


class Chunker:
    """Cuts streamed LLM text into sentences for TTS.

    The first chunk may also end at a comma (once it has a few characters) so speech starts earlier.
    """

    END = "。！？!?；;\n"
    SOFT = "，,、"

    def __init__(self, first_min: int = 6):
        self.buf = ""
        self.first = True
        self.first_min = first_min

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while (i := self._cut()) is not None:
            piece, self.buf = self.buf[: i + 1], self.buf[i + 1:]
            piece = clean(piece)
            if re.search(r"\w", piece):
                out.append(piece)
                self.first = False
        return out

    def flush(self) -> list[str]:
        rest, self.buf = clean(self.buf), ""
        return [rest] if re.search(r"\w", rest) else []

    def _cut(self) -> int | None:
        for i, ch in enumerate(self.buf):
            if ch in self.END or (self.first and ch in self.SOFT and i + 1 >= self.first_min):
                return i
        return None


# ---------------------------------------------------------------- microphone

def rms(frame: bytes) -> float:
    samples = array("h", frame)
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


def to_wav(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MIC_RATE)
        w.writeframes(pcm)
    return out.getvalue()


class Mic:
    """Continuously captures 30 ms frames of 16 kHz mono PCM into an asyncio queue."""

    def __init__(self, device, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.q: asyncio.Queue[bytes] = asyncio.Queue()
        self.stream = sd.RawInputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                                        blocksize=FRAME_SAMPLES, device=device, callback=self._callback)

    def _callback(self, indata, frames, time_info, status) -> None:
        self.loop.call_soon_threadsafe(self.q.put_nowait, bytes(indata))

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()

    def clear(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()

    async def frame(self) -> bytes:
        return await self.q.get()


async def calibrate(mic: Mic, seconds: float = 0.6) -> float:
    """Median ambient level, used to pick the speech threshold."""
    levels = sorted([rms(await mic.frame()) for _ in range(int(seconds * 1000 / FRAME_MS))])
    return levels[len(levels) // 2]


async def listen(mic: Mic, threshold: float, silence_ms: int, max_seconds: float):
    """Wait for one utterance. Returns (wav_bytes, time_speech_ended) or None if it was just noise."""
    preroll: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
    voiced_run = 0
    buf: bytearray | None = None
    loud_frames = silent_run = 0

    while True:
        frame = await mic.frame()
        loud = rms(frame) > threshold
        if buf is None:
            preroll.append(frame)
            voiced_run = voiced_run + 1 if loud else 0
            if voiced_run >= START_FRAMES:
                buf = bytearray(b"".join(preroll))
                loud_frames, silent_run = voiced_run, 0
            continue
        buf += frame
        if loud:
            loud_frames += 1
            silent_run = 0
        else:
            silent_run += 1
        if silent_run * FRAME_MS >= silence_ms or len(buf) >= max_seconds * MIC_RATE * 2:
            break

    ended = time.perf_counter() - silent_run * FRAME_MS / 1000
    if loud_frames < MIN_LOUD_FRAMES:
        return None
    keep_silence = 7  # ~200 ms of trailing silence is enough
    del buf[len(buf) - max(0, silent_run - keep_silence) * FRAME_BYTES:]
    return to_wav(bytes(buf)), ended


# ------------------------------------------------------------------- speaker

class Speaker:
    """Plays 16-bit mono PCM from a queue on its own thread (stream.write blocks)."""

    def __init__(self, rate: int, device):
        self.stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16",
                                         device=device, latency="low")
        self.q: queue.Queue[bytes | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.stream.start()
        self.thread.start()

    def stop(self) -> None:
        self.clear()
        self.q.put(None)
        self.stream.stop()
        self.stream.close()

    def _run(self) -> None:
        while (data := self.q.get()) is not None:
            try:
                self.stream.write(data)
            finally:
                self.q.task_done()

    def play(self, data: bytes) -> None:
        self.q.put(data)

    def clear(self) -> None:
        try:
            while True:
                self.q.get_nowait()
                self.q.task_done()
        except queue.Empty:
            pass

    async def drain(self) -> None:
        await asyncio.to_thread(self.q.join)


# ----------------------------------------------------------------------- LLM

def stream_llm(messages: list[dict], args: argparse.Namespace, api_key: str,
               out: asyncio.Queue, loop: asyncio.AbstractEventLoop, cancel: threading.Event) -> None:
    """Worker thread: pushes text deltas onto `out`, then None when done (or an Exception)."""

    def put(item) -> None:
        loop.call_soon_threadsafe(out.put_nowait, item)

    models = [m.strip() for m in args.llm.split(",") if m.strip()]
    # one model -> "model"; several -> OpenRouter's "models" list, tried in order
    model_field = {"model": models[0]} if len(models) == 1 else {"models": models}
    # Reasoning models think before the first word, which is dead air in a voice chat.
    # Models without reasoning ignore the field.
    reasoning = {} if args.reasoning_effort == "omit" else {"reasoning": {"effort": args.reasoning_effort}}

    try:
        with requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={**model_field, **reasoning, "provider": {"sort": "latency"},  # fastest first token
                  "messages": messages, "stream": True,
                  "max_tokens": args.max_tokens, "temperature": 0.7},
            stream=True,
            timeout=(5, 60),
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text}")
            for raw in resp.iter_lines(chunk_size=None):
                if cancel.is_set():
                    break
                line = raw.decode("utf-8")
                if not line.startswith("data:"):  # blank keep-alives and ": OPENROUTER PROCESSING"
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("error"):
                    raise RuntimeError(f"OpenRouter error: {chunk['error']}")
                delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if delta:
                    put(delta)
        put(None)
    except Exception as e:  # noqa: BLE001 - forwarded to the asyncio side
        put(e)


# ----------------------------------------------------------------------- TTS

class TTSStream:
    """One ElevenLabs text-to-dialogue WebSocket per reply (same protocol as realtime_tts.py)."""

    def __init__(self, args: argparse.Namespace, api_key: str, speaker: Speaker):
        self.args, self.api_key, self.speaker = args, api_key, speaker
        self.ws = None
        self.receiver: asyncio.Task | None = None
        self.first_audio_at: float | None = None

    async def start(self) -> None:
        query = urlencode({
            "model_id": self.args.model,
            "language_code": self.args.language,
            "output_format": f"pcm_{self.args.tts_rate}",
        })
        self.ws = await websockets.connect(
            f"{realtime_tts.HOST}/v1/text-to-dialogue/stream-input?{query}",
            additional_headers={"xi-api-key": self.api_key},
        )
        await self.ws.send(json.dumps({"voices": [self.args.voice]}))
        self.receiver = asyncio.create_task(self._receive())

    async def say(self, text: str) -> None:
        await self.ws.send(json.dumps({
            "inputs": [{"text": text + " ", "voice_id": self.args.voice, "new_turn": False}],
            "flush": True,  # generate now instead of waiting for the server's buffer threshold
        }))

    async def finish(self) -> None:
        await self.ws.send(json.dumps({"close_socket": True}))
        await self.receiver

    async def abort(self) -> None:
        if self.receiver:
            self.receiver.cancel()
            await asyncio.gather(self.receiver, return_exceptions=True)
        if self.ws:
            await self.ws.close()

    async def _receive(self) -> None:
        carry = b""
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg.get("error"):
                raise RuntimeError(f"ElevenLabs {msg['error']}: {msg.get('message')}")
            if msg.get("audio"):
                if self.first_audio_at is None:
                    self.first_audio_at = time.perf_counter()
                chunk = carry + base64.b64decode(msg["audio"])
                carry = chunk[-1:] if len(chunk) % 2 else b""  # keep 16-bit samples aligned
                self.speaker.play(chunk[: len(chunk) - len(carry)])
            if msg.get("is_final"):
                return


# --------------------------------------------------------------------- agent

class Agent:
    def __init__(self, args: argparse.Namespace, eleven_key: str, openrouter_key: str, speaker: Speaker):
        self.args, self.eleven_key, self.openrouter_key, self.speaker = args, eleven_key, openrouter_key, speaker
        self.history: list[dict] = []

    async def respond(self, user_text: str, speech_ended: float, stt_ms: float) -> None:
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.args.system}] + self.history[-self.args.history:]

        loop = asyncio.get_running_loop()
        deltas: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()
        threading.Thread(target=stream_llm, daemon=True,
                         args=(messages, self.args, self.openrouter_key, deltas, loop, cancel)).start()
        t_llm = time.perf_counter()

        tts = TTSStream(self.args, self.eleven_key, self.speaker)
        tts_ready = asyncio.create_task(tts.start())  # connect while the LLM is still thinking
        chunker, reply = Chunker(), []
        first_sentence_ms = None

        try:
            while True:
                item = await deltas.get()
                if isinstance(item, Exception):
                    raise item
                for piece in (chunker.flush() if item is None else chunker.feed(item)):
                    if first_sentence_ms is None:
                        first_sentence_ms = (time.perf_counter() - t_llm) * 1000
                    await tts_ready
                    await tts.say(piece)
                    reply.append(piece)
                    print(piece, end="", flush=True)
                if item is None:
                    break
            await tts_ready
            await tts.finish()
            await self.speaker.drain()
        except BaseException:
            cancel.set()
            tts_ready.cancel()
            await asyncio.gather(tts_ready, return_exceptions=True)
            await tts.abort()
            self.speaker.clear()
            raise
        finally:
            cancel.set()
            if reply:
                self.history.append({"role": "assistant", "content": "".join(reply)})
            else:
                self.history.pop()  # keep user/assistant turns paired

        if tts.first_audio_at is not None:
            print(f"\n[stt {stt_ms:.0f} ms | llm first sentence {first_sentence_ms or 0:.0f} ms | "
                  f"first audio {(tts.first_audio_at - speech_ended) * 1000:.0f} ms after you stopped talking]",
                  file=sys.stderr)


# ---------------------------------------------------------------------- main

async def run(args: argparse.Namespace) -> None:
    eleven_key = os.environ.get("ELEVENLABS_API_KEY") or sys.exit("Set ELEVENLABS_API_KEY first.")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("Set OPENROUTER_API_KEY first.")

    loop = asyncio.get_running_loop()
    speaker = Speaker(args.tts_rate, args.output_device)
    mic = Mic(args.input_device, loop)
    speaker.start()
    mic.start()
    try:
        print("Calibrating microphone, stay quiet for a moment...", file=sys.stderr)
        ambient = await calibrate(mic)
        threshold = args.threshold or max(ambient * 3, 250.0)
        print(f"[ambient level {ambient:.0f}, speech threshold {threshold:.0f}; "
              f"use --threshold to change]", file=sys.stderr)

        agent = Agent(args, eleven_key, openrouter_key, speaker)
        while True:
            mic.clear()  # drop anything captured while the agent was talking
            print("\n[listening...]", file=sys.stderr)
            heard = await listen(mic, threshold, args.silence_ms, args.max_seconds)
            if heard is None:
                continue
            wav, speech_ended = heard

            t0 = time.perf_counter()
            try:
                text = await asyncio.to_thread(transcribe.transcribe_bytes, wav, eleven_key,
                                               "speech.wav", args.stt_model, args.language)
            except Exception as e:  # noqa: BLE001
                print(f"[speech-to-text failed: {e}]", file=sys.stderr)
                continue
            stt_ms = (time.perf_counter() - t0) * 1000
            if not text:
                print("[didn't catch that]", file=sys.stderr)
                continue

            print(f"you:   {text}")
            print("agent: ", end="", flush=True)
            try:
                await agent.respond(text, speech_ended, stt_ms)
            except Exception as e:  # noqa: BLE001
                print(f"\n[reply failed: {e}]", file=sys.stderr)
            await asyncio.sleep(0.3)  # let the speaker tail die out before the mic is read again
    finally:
        mic.stop()
        speaker.stop()


def device(value: str | None):
    return None if value is None else int(value) if value.isdigit() else value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voice", default=os.environ.get("ELEVENLABS_VOICE_ID", realtime_tts.DEFAULT_VOICE),
                   help="ElevenLabs voice ID (default: $ELEVENLABS_VOICE_ID); use a Cantonese voice")
    p.add_argument("--llm", default=DEFAULT_LLM,
                   help="OpenRouter chat model, or comma-separated fallbacks (default: %(default)s)")
    p.add_argument("--system", default=SYSTEM_PROMPT, help="system prompt (default: short colloquial Cantonese)")
    p.add_argument("--max-tokens", type=int, default=1000,
                   help="token cap incl. reasoning tokens (default: 1000); reply length is set by the prompt")
    p.add_argument("--reasoning-effort", default="none",
                   choices=["none", "minimal", "low", "medium", "high", "xhigh", "omit"],
                   help="reasoning effort (default: none = fastest). Models that can't disable "
                        "reasoning, e.g. grok-4.7, reject 'none': use 'low' or 'omit' with them")
    p.add_argument("--history", type=int, default=20, help="messages of context to keep (default: 20)")
    p.add_argument("--language", default="yue", help="language_code for STT and TTS (default: yue)")
    p.add_argument("--model", default=realtime_tts.DEFAULT_MODEL, help="TTS model (default: %(default)s)")
    p.add_argument("--stt-model", default=transcribe.DEFAULT_MODEL, help="STT model (default: %(default)s)")
    p.add_argument("--tts-rate", type=int, default=24000, choices=[16000, 22050, 24000, 44100],
                   help="TTS PCM sample rate (default: 24000)")
    p.add_argument("--silence-ms", type=int, default=700,
                   help="silence that ends your turn (default: 700; lower = faster but may cut you off)")
    p.add_argument("--threshold", type=float, default=None,
                   help="mic RMS level that counts as speech (default: auto from ambient noise)")
    p.add_argument("--max-seconds", type=float, default=30, help="longest single utterance (default: 30)")
    p.add_argument("--input-device", help="microphone index or name (see --list-devices)")
    p.add_argument("--output-device", help="speaker index or name (see --list-devices)")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    args.input_device, args.output_device = device(args.input_device), device(args.output_device)
    if args.voice == realtime_tts.DEFAULT_VOICE:
        print("[warning: default voice is not a native Cantonese voice; "
              "run realtime_tts.py --list-voices and pass --voice]", file=sys.stderr)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nbye", file=sys.stderr)


if __name__ == "__main__":
    main()
