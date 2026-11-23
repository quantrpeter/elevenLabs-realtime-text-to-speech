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
- Optional: `OPENROUTER_API_KEY`, only for `--rewrite`

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

## Notes

- Each utterance is sent with `flush`, so audio is generated immediately instead of waiting for the server's buffering threshold.
- The connection stays open in interactive mode; a keep-alive is sent every 10 s because the server closes idle sessions after 20 s.
- With `eleven_v4_turbo`, only one voice can be registered per connection.
