## Vshift
a voice to tts tool for linux
## Install

Use Python 3.13 or 3.12 for the Whisper environment. The included installer prefers 3.13.

```bash
./install.sh
```

On Arch/CachyOS, install PortAudio first when necessary:

```bash
sudo pacman -S --needed portaudio
```

Then:

```bash
export PATH="$HOME/.local/bin:$PATH"
vshift discord
```

The installer creates a fresh `venv`, so it never reuses an older environment.

On CachyOS/Arch the system `python` is usually newer than 3.13. Install an older one
(`yay -S python313`, or `uv python install 3.13`) and, if the installer doesn't find
it on its own, point it there:

```bash
VSHIFT_PYTHON=/path/to/python3.13 ./install.sh
```

## CLI

Useful commands:

```bash
vshift --help
vshift --version
vshift doctor
vshift config
vshift config --json
vshift config --reset
vshift mics
vshift mics --json
vshift devices
vshift devices --json
vshift voices
vshift voices --installed
vshift download en_US-amy-medium
vshift download en_US-amy-medium --force   # re-fetch a damaged voice
vshift cleanup
```

`vshift cleanup` removes only the default vshift virtual mic/sink and any stale vshift monitoring loopback. This is useful after an interrupted session.

`vshift --help`, `vshift --version`, `vshift config`, and `vshift voices` do not require the audio runtime to be initialized.

## Discord

Run:

```bash
vshift discord
```

Choose `vshift_mic_input` as the Discord input device.

Controls:

- `M` — choose the microphone vshift listens to (see "Choosing your microphone" above)
- `H` — toggle local monitoring of the **generated** voice (off by default — see above)
- `[` / `]` — cycle voices
- `V` — browse, search and preview voices (see "Voice browser" above)
- `L` — translate what you say into English (see "Translation" above)
- `T` — type text and have it spoken through the current voice, no mic needed
- `Space` — pause/resume
- `↑` / `↓` — adjust speech threshold (watch the `Mic level` bar; `┃` is the threshold)
- `C` — auto-calibrate the threshold: stay quiet for 2 seconds
- `←` / `→` — adjust silence timeout
- `-` / `+` — speak slower / faster; `0` resets to the voice default
- `,` / `.` — quieter / louder (also `<` / `>`)
- `R` — repeat the last line
- `Q` — quit

A voice switch is queued and loaded by the isolated Piper worker. It takes effect for the next synthesized utterance and does not require restarting vshift.

## Type-to-speak (`T`)

Press `T` to open a prompt at the bottom of the screen, type anything, hit
Enter. It's synthesized in your current voice and played out the same route
as converted speech (virtual mic → Discord/games, or your speakers) — no mic
input involved at all. Useful for saying something precisely, saying
something without making noise, or just testing a voice.

It runs through the exact same TTS/playback stages as your spoken voice
(including the sentence-by-sentence streaming above), so a typed paragraph
starts playing its first sentence while later ones are still synthesizing,
same as speaking would. It also counts toward "Utterances" and shows up in
"Last heard" prefixed with `(typed)` so it's clear where it came from.
The mic is briefly paused while the prompt is open (same as the existing `V`
voice-id prompt) and resumes to whatever it was before once you hit
Enter/cancel.

## Accurate status

The status line used to be a single string that whichever pipeline stage
ran last would overwrite — which could straight-up lie: pausing set it to
`PAUSED`, but nothing ever set it back on unpause unless STT/TTS/playback
happened to fire again first, so it could get stuck reading "Paused" while
actively listening. It also had no way to represent the pipeline being
genuinely concurrent (Whisper transcribing what you just said while Piper is
still speaking the previous reply).

Status is now computed fresh on every redraw from the actual state of each
stage - `paused`, `error`, `stt_busy`, `tts_busy`, `playing` - rather than
being a value stages push and hope stays correct:

- **`Listening`** — idle, nothing in flight
- **`Thinking`** — STT transcribing
- **`Speaking`** — TTS synthesizing and/or playback in progress
- **`Speaking + Thinking`** — genuinely both at once (the common case once
  you talk again while a reply is still playing)
- **`(+N queued)`** appended whenever there's a backlog waiting anywhere in
  the pipeline, so you can see when you're talking faster than it can keep up
- **`Paused`** / **`Error — retrying`** — take priority over everything else,
  and clear immediately and correctly when the underlying condition does

## Crash diagnostics

Piper worker stderr is written to:

```text
~/.cache/vshift/tts_worker.log
```

Whisper worker failures are reported in the UI and the worker is restarted automatically.

The application still may report a TTS worker failure in the UI if Piper itself aborts, but the main `vshift` process should remain alive.

## Known upstream context

Piper's current Python API loads an ONNX voice with `PiperVoice.load()` and supports streaming synthesis. The upstream project also publishes Linux amd64/arm64 binary releases. The Python implementation uses the API inside its dedicated process rather than importing Piper into the parent process.


## Diagnostics

STT and TTS worker stderr are logged to:

```text
~/.cache/vshift/stt_worker.log
~/.cache/vshift/tts_worker.log
```

Settings are stored at `~/.config/vshift/settings.json` and written atomically.

