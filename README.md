# ElevenLabs Realtime Text-to-Speech (Cantonese)

Low-latency Cantonese text-to-speech over the ElevenLabs WebSocket API.

- Model: `eleven_v4_turbo` (~100 ms median latency, supports Cantonese `yue`)
- Endpoint: `wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input` (v4 models are only served by the Text-to-Dialogue WebSocket, not the classic TTS `stream-input`)
- Audio is requested as raw PCM and piped to a player chunk by chunk, so playback starts as soon as the first chunk arrives
- Prints time to first audio for every utterance

## Requirements

- Python 3.10+
- An ElevenLabs API key (`ELEVENLABS_API_KEY`) with Text to Speech access. `--list-voices` also needs **Voices → Read**
- `ffplay` (from ffmpeg) or `sox` for playback (not needed with `--no-play`)
- Optional: `OPENROUTER_API_KEY`, only for `--rewrite` in `realtime_tts.py`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ELEVENLABS_API_KEY=sk_...
```

## Usage

```bash
# one-shot
python realtime_tts.py --voice <voice_id> "你好，歡迎使用即時語音合成。"

# interactive: one line = one utterance, Ctrl-D to quit
python realtime_tts.py --voice <voice_id>

# also save to a WAV file
python realtime_tts.py --voice <voice_id> "今日天氣好好" -o out.wav

# synthesize without playing
python realtime_tts.py --voice <voice_id> "你好" --no-play -o out.wav
```

### Keeping the output Cantonese

The script always sends `language_code=yue`, but the model treats it as a hint. Standard written Chinese with no Cantonese wording can still be read as Mandarin. Two ways to prevent that:

1. **Use a Cantonese voice** (strongest effect, no extra latency). The voice's accent drives the output.
   ```bash
   python realtime_tts.py --list-voices          # lists library voices for --language (default: yue)
   python realtime_tts.py --voice <voice_id> "..."
   ```
   Library voices may need to be added to **My Voices** in your account before the API accepts them. The default voice is not Cantonese, so the script warns when you use it.
2. **Rewrite into colloquial Cantonese** with `--rewrite`. Text is converted (係, 唔, 嘅, 咗, 喺…) via OpenRouter before synthesis. This works with any voice but adds an LLM round trip.
   ```bash
   export OPENROUTER_API_KEY=sk-or-...
   python realtime_tts.py --voice <voice_id> --rewrite "你現在在哪裡？"
   ```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `text` | – | Text to speak. Omit for interactive mode |
| `--voice` | Rachel (`21m00Tcm4TlvDq8ikWAM`) | ElevenLabs voice ID |
| `--list-voices` | – | List library voices for `--language` and exit |
| `--language` | `yue` | `language_code` sent to the API |
| `--model` | `eleven_v4_turbo` | Any `eleven_v3*` / `eleven_v4*` model. `eleven_v4` has the highest quality but is slower |
| `--rewrite` | off | Rewrite text into colloquial Cantonese first |
| `--rewrite-model` | `google/gemini-2.5-flash-lite` | OpenRouter model used by `--rewrite` |
| `--sample-rate` | `24000` | PCM rate: 16000, 22050, 24000 or 44100 |
| `-o`, `--output` | – | Also save the audio as a WAV file |
| `--no-play` | off | Don't play audio |

## Audio to text (Cantonese)

[transcribe.py](transcribe.py) transcribes an audio or video file with ElevenLabs Scribe (`scribe_v3`), with the language forced to Cantonese (`yue`) instead of auto-detected.

Scribe returns **Simplified Chinese** (it has no option to change this), so the script converts the result to **Traditional Chinese** (Hong Kong variants) with OpenCC. This is offline and deterministic. Scribe may also return written-style wording rather than colloquial Cantonese; the script does not alter the wording.

```bash
python transcribe.py recording.mp3
python transcribe.py recording.wav -o transcript.txt
python transcribe.py meeting.m4a --diarize              # one line per speaker turn
python transcribe.py clip.mp3 --json                    # full API response with word timestamps
python transcribe.py clip.mp3 --keyterm 港鐵 --keyterm 八達通   # bias toward specific words (+20% cost)
```

Options: `--model`, `--language` (default `yue`), `--diarize`, `--audio-events`, `--keyterm` (repeatable), `--no-traditional`, `--json`, `-o/--output`. It needs `ELEVENLABS_API_KEY` with Speech to Text access.

## Voice agent (mic in, speaker out)

[agent.py](agent.py) is a Cantonese voice assistant: you speak into the microphone and it answers out loud.

```
mic → silence detection → Scribe (transcribe.py) → OpenRouter LLM (streamed)
    → sentence chunks → ElevenLabs realtime TTS (realtime_tts.py) → speaker
```

```bash
export ELEVENLABS_API_KEY=sk_...
export OPENROUTER_API_KEY=sk-or-...
python agent.py --list-devices
python agent.py --voice <cantonese_voice_id>
python agent.py --voice <id> --llm x-ai/grok-4.3 --input-device 0 --output-device 1
```

The default model is `x-ai/grok-4.3` with reasoning turned off, because reasoning is dead air in a voice conversation. Requests also ask OpenRouter for the lowest-latency provider (`provider.sort = latency`). `x-ai/grok-4.7` was tried and felt slow: its reasoning can't be disabled, so it always thinks first. If you still want it, use `--llm x-ai/grok-4.7 --reasoning-effort low`. `x-ai/grok-4.20` (reasoning off by default) and `deepseek/deepseek-chat` are other fast options. Compare them with the `llm first sentence` time printed after each reply.

If OpenRouter replies `403 ... not available in your region`, that model's providers are geo-blocked for you. Pick another model; `--llm a/b,c/d` tries several in order.

Set `ELEVENLABS_VOICE_ID` to avoid passing `--voice` every time. Press Ctrl-C to quit. On macOS the terminal needs microphone permission (System Settings → Privacy & Security → Microphone).

How it keeps latency low:
- The TTS WebSocket is opened while the LLM is still thinking, and each sentence is sent to TTS (and starts playing) as soon as the LLM has written it. The first sentence may end at a comma so speech starts sooner.
- After every reply it prints a timing line: speech-to-text time, LLM time to first sentence, and time to first audio measured from when you stopped talking.

Behaviour to know about:
- **No interruption:** the mic is ignored while the agent speaks, so it doesn't hear itself. Wait for it to finish; keep replies short (the default prompt asks for 1–3 sentences).
- **End of turn:** a turn ends after `--silence-ms` (default 700) of silence. Lower it for snappier replies, raise it if you get cut off. The speech threshold is calibrated from the first 0.6 s of room noise, so stay quiet at startup, or set `--threshold`.
- **LLM output:** the system prompt asks for colloquial Cantonese in Traditional Chinese with no markdown or emoji. Replacing it with `--system` is possible. The LLM output is also cleaned and converted to Traditional (Hong Kong) before being spoken.
- **Context:** the last 20 messages are kept (`--history`).

| Option | Default | Description |
| --- | --- | --- |
| `--voice` | `$ELEVENLABS_VOICE_ID`, else Rachel | ElevenLabs voice ID |
| `--llm` | `x-ai/grok-4.3` | OpenRouter chat model(s), comma-separated for fallback |
| `--reasoning-effort` | `none` | Reasoning effort (`omit` sends nothing; `grok-4.7` needs `low` or `omit`) |
| `--system` | short colloquial Cantonese prompt | System prompt |
| `--max-tokens` | `1000` | Token cap, including reasoning tokens |
| `--history` | `20` | Messages of context kept |
| `--language` | `yue` | `language_code` for STT and TTS |
| `--model` / `--stt-model` | `eleven_v4_turbo` / `scribe_v3` | TTS and STT models |
| `--tts-rate` | `24000` | TTS PCM sample rate |
| `--silence-ms` | `700` | Silence that ends your turn |
| `--threshold` | auto | Mic RMS level that counts as speech |
| `--max-seconds` | `30` | Longest single utterance |
| `--input-device` / `--output-device` | system default | Index or name from `--list-devices` |

## Notes

- Each utterance is sent with `flush`, so audio is generated immediately instead of waiting for the server's buffering threshold.
- The connection stays open in interactive mode; a keep-alive is sent every 10 s because the server closes idle sessions after 20 s.
- With `eleven_v4_turbo`, only one voice can be registered per connection.
