#!/usr/bin/env python3
"""Cantonese audio-to-text (speech-to-text) with ElevenLabs Scribe.

Transcribes an audio or video file, forcing the language to Cantonese (`yue`)
so the model doesn't auto-detect Mandarin. Scribe returns Simplified Chinese, so the
text is then converted to Traditional (Hong Kong) with OpenCC.

Usage:
    export ELEVENLABS_API_KEY=sk_...
    python transcribe.py recording.mp3
    python transcribe.py recording.wav -o transcript.txt
    python transcribe.py meeting.m4a --diarize          # label speakers
    python transcribe.py clip.mp3 --json                # full API response
    python transcribe.py out.wav --keyterm 港鐵 --keyterm 八達通
"""

import argparse
import io
import json
import os
import sys
import time

import requests
from opencc import OpenCC

URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_MODEL = "scribe_v3"  # most accurate model for recorded audio


def scribe_request(fileobj, filename: str, api_key: str, data: list[tuple[str, str]],
                   timeout: tuple[int, int] = (10, 600)) -> dict:
    """POST one file to Scribe; raises RuntimeError on a non-200 response."""
    resp = requests.post(
        URL,
        headers={"xi-api-key": api_key},
        data=data,
        files={"file": (filename, fileobj)},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def transcribe(args: argparse.Namespace, api_key: str) -> dict:
    data: list[tuple[str, str]] = [
        ("model_id", args.model),
        ("language_code", args.language),  # forced; leaving it unset triggers auto-detection
        ("tag_audio_events", "true" if args.audio_events else "false"),
        ("diarize", "true" if args.diarize else "false"),
        ("timestamps_granularity", "word" if (args.diarize or args.json) else "none"),
    ]
    data += [("keyterms", k) for k in args.keyterm or []]

    try:
        with open(args.audio, "rb") as f:
            return scribe_request(f, os.path.basename(args.audio), api_key, data)
    except RuntimeError as e:
        sys.exit(str(e))


def transcribe_bytes(audio: bytes, api_key: str, filename: str = "speech.wav",
                     model: str = DEFAULT_MODEL, language: str = "yue") -> str:
    """Transcribe in-memory audio (e.g. a WAV from the mic) to Traditional Chinese text."""
    data = [
        ("model_id", model),
        ("language_code", language),
        ("tag_audio_events", "false"),
        ("diarize", "false"),
        ("timestamps_granularity", "none"),
    ]
    result = scribe_request(io.BytesIO(audio), filename, api_key, data, timeout=(10, 60))
    return OpenCC("s2hk").convert(result.get("text", "")).strip()


def to_traditional(result: dict) -> None:
    """Scribe returns Simplified Chinese; convert in place to Traditional (Hong Kong variants)."""
    cc = OpenCC("s2hk")
    result["text"] = cc.convert(result.get("text", ""))
    for w in result.get("words", []):
        w["text"] = cc.convert(w.get("text", ""))


def format_text(result: dict, diarize: bool) -> str:
    """Plain text; with --diarize, one line per speaker turn."""
    if not diarize:
        return result.get("text", "").strip()

    lines: list[str] = []
    speaker = None
    for w in result.get("words", []):
        if w.get("type") == "audio_event":
            continue
        if w.get("speaker_id") != speaker:
            speaker = w.get("speaker_id")
            lines.append(f"{speaker}: ")
        lines[-1] += w.get("text", "")
    return "\n".join(line.strip() for line in lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", help="audio or video file (mp3, wav, m4a, mp4, ...)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    p.add_argument("--language", default="yue", help="language_code (default: yue = Cantonese)")
    p.add_argument("--diarize", action="store_true", help="label who is speaking")
    p.add_argument("--audio-events", action="store_true", help="tag non-speech sounds, e.g. (laughter)")
    p.add_argument("--keyterm", action="append", metavar="TERM",
                   help="bias recognition toward a word/phrase; repeatable (adds ~20%% cost)")
    p.add_argument("--no-traditional", action="store_true",
                   help="keep Scribe's Simplified Chinese (default: convert to Traditional, HK)")
    p.add_argument("--json", action="store_true", help="print the full API response as JSON")
    p.add_argument("-o", "--output", help="also write the result to this file")
    args = p.parse_args()

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit("Set ELEVENLABS_API_KEY first.")
    if not os.path.isfile(args.audio):
        sys.exit(f"File not found: {args.audio}")

    t0 = time.perf_counter()
    result = transcribe(args, api_key)
    print(f"[transcribed in {time.perf_counter() - t0:.1f} s]", file=sys.stderr)

    if not args.no_traditional:
        to_traditional(result)

    output = (json.dumps(result, ensure_ascii=False, indent=2) if args.json
              else format_text(result, args.diarize))
    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"saved {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
