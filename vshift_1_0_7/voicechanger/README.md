# vshift 1.0.7

A terminal voice changer for Linux: microphone → faster-whisper → Piper → virtual microphone.

## Architecture and crash isolation

vshift isolates **both** Whisper and Piper/ONNX Runtime in child processes. The parent process handles only the microphone, UI, and a single PortAudio output stream. This is deliberate: native runtime failures such as `abort (core dumped)` / glibc heap-corruption messages cannot take down the UI/audio-control process.

The generated voice is written to the virtual sink used by Discord. Local monitoring is done with a PipeWire/PulseAudio `module-loopback`, not with a second simultaneous PortAudio output stream. That removes the concurrent-output path used by earlier builds.

Whisper is also configured to be less willing to turn silence/noise into repeated short hallucinations. Short repeated `OK`/`okay` results are discarded before Piper is called, and segments whose own confidence/compression-ratio scores flag them as garbage are dropped before they ever reach Piper.

**Local monitoring (`H`) is off by default.** Looping the generated voice back to your real speakers while the mic is actively listening is a genuine feedback risk unless you're on headphones — the mic can pick the speech back up, which then gets transcribed and spoken again, and again. Turn it on with `H` or `--monitor` once you've confirmed your setup (e.g. headphones, or speakers/mic far enough apart) won't just hear itself.

As a second line of defense, whenever an acoustic path back to the mic exists (monitoring on, or `vshift run` playing on real speakers), vshift remembers what it just said for a few seconds and discards anything the mic hears that matches it — so a stray echo doesn't turn into a runaway repeat loop. Short phrases only count as echoes when they are identical; longer ones must be a close match. With monitoring off in `vshift discord` there is no path from the generated voice to your mic, so the guard is off and never eats anything you say.

## Pipelined listening (lower perceived latency)

Continuous mode now runs as **4 concurrent stages** connected by queues instead of one blocking loop:

```
capture+VAD  →  STT (Whisper)  →  TTS (Piper)  →  playback
 (mic never       (own subprocess)  (own subprocess)  (own thread)
  closes)
```

Previously the mic stream was opened, closed, and reopened around every single
utterance — while a reply was being transcribed, synthesized, played, and
followed by a cooldown, the mic was simply off, and anything said in that
window was gone. Now `capture_loop` owns the mic for the whole session and
keeps segmenting speech the entire time, including while a previous reply is
still being processed or spoken out loud. Each captured utterance is queued
and worked through the pipeline independently, so:

- Talking again while vshift is still speaking your previous line no longer
  gets lost — it's captured and processed as soon as the pipeline is free.
- Because STT and TTS run in *separate* subprocesses, utterance N+1 can be
  transcribing while utterance N is still being synthesized or played —
  real overlap, not just "not dropped."

Trade-offs / things to know:

- **Throughput, not raw per-utterance speed.** A single utterance still takes
  roughly STT-time + TTS-time to come back; what's fixed is that talking
  again during that window is no longer wasted.
- **If you talk continuously without pausing**, utterances will queue up
  faster than a single STT+TTS pair can drain them, and replies will play
  back in order with growing delay — same as a person who can't get a word
  in. This is inherent to any turn-based STT→TTS design, not something a
  faster machine fully removes.
- **Echo guard applies when monitoring is on.** With monitoring (`H`/`--monitor`)
  on, or real speakers near the mic, the mic can pick up playback while it's
  still listening. vshift discards close matches to what it just said (see
  above), but if you use monitoring, prefer headphones anyway.

### Further reducing latency

None of these are on by default because they trade something for speed —
pick what you don't mind giving up:

| Lever | Effect | Cost |
|---|---|---|
| `--stt-model tiny.en` (instead of `base.en`) | Faster transcription | Less accurate, more misheard words |
| Lower `--silence-ms` (e.g. `400`) | Ends your utterance sooner after you pause | More likely to cut you off mid-sentence if you pause to think |
| Raise `↑`/`--threshold` a bit if your mic is noisy | Fewer false utterance-starts from background noise re-triggering the pipeline | Needs a clear voice level; too high and quiet speech gets missed |

The biggest remaining latency cost is genuinely STT+TTS compute time on CPU —
`tiny.en` is the single largest lever if you need it snappier and can live
with the accuracy hit.

## Sentence-level TTS streaming

Multi-sentence utterances now stream: the transcribed text is split into
sentences (`split_into_sentences`), and each sentence becomes its own item
in the STT→TTS→playback pipeline instead of one Piper call for the whole
utterance. The practical effect: if you say two or three sentences in one
breath, the first one starts playing as soon as *it's* synthesized, while
the next sentence is still being synthesized — instead of everything
waiting on one big Piper call for the full utterance before any sound plays.
It's a lightweight regex split on `. ! ?` with a small guard against
common abbreviations ("Dr.", "e.g.") so those don't cause a false split; it
doesn't need to be linguistically perfect, since a slightly-off split just
moves where a pause falls, whereas not splitting at all is the behavior
this replaces.

### Other optimizations looked at, not applied

Looked at these and decided the effort/risk wasn't worth it *yet* relative
to what they'd save — noting them here in case you want to push further:

- **True IPC-level audio streaming.** Piper's `voice.synthesize()` already
  yields audio in chunks internally, but `tts_worker.py` collects every
  chunk and sends back one complete blob per request. Streaming those chunks
  back progressively (multiple `audio_chunk` messages instead of one) would
  give sub-sentence granularity on top of what sentence-splitting already
  does. Skipped for now because it means changing the wire protocol in two
  files that have to stay in lockstep (`tts_worker.py` and `TTSClient`), and
  I can't fully integration-test that against real Piper models in this
  environment — sentence-splitting gets most of the same benefit for your
  actual usage pattern (converting your own speech, which is usually a few
  short sentences, not long paragraphs) at much lower risk.
- **Base64 → raw binary framing for TTS audio.** The STT leg already sends
  audio as raw length-prefixed bytes (`stt_worker.py`); the TTS leg still
  base64-encodes its response. Switching it to match would shave a small
  amount of CPU/latency (base64 costs ~33% size + an encode/decode pass) but
  for a few seconds of int16 audio that's low milliseconds — real but minor,
  and again a two-file protocol change I'd want to test more thoroughly
  before shipping.
- **`beam_size`/`cpu_threads` tuning.** `stt_worker.py` uses `beam_size=3` as
  a reasonable middle ground; dropping to `1` (greedy decoding) is faster at
  some accuracy cost. Neither is exposed as a flag yet - hardcoding a
  "faster" default isn't safe since the right value depends on your CPU core
  count and how much you value accuracy, so this is better as a flag you set
  than a default I guess for you.
- **Disabling faster-whisper's internal `vad_filter`.** `stt_worker.py`
  already runs a second (Silero) VAD pass inside Whisper on top of vshift's
  own energy-based segmenter, which costs a little time. Left it on
  deliberately - it's doing real work trimming pauses inside your own
  segmented audio, and it's tied into the anti-hallucination tuning
  (`compression_ratio_threshold`, `no_speech_threshold`) that was added to
  fix the repeated-phrase bug. Touching it risks reintroducing that.

## Unreleased: voice browser, translation, fixes and a refreshed panel

### Choosing your microphone (`M`, `--mic`, `vshift mics`)

vshift used to listen to whatever your system default microphone was. Now you choose:

```bash
vshift mics                        # list microphones (● = saved choice)
vshift discord --mic fifine        # any part of a mic's name; saved for next time
vshift discord --mic default       # back to the system default
```

- In the panel, `M` opens a list of your microphones; pick one and vshift switches to it
  immediately, without restarting. The new `Microphone` row shows which one it's using.
- Mics are listed by their PipeWire/PulseAudio name ("fifine Microphone Mono"). Real
  mics come first, virtual ones are marked, and speaker monitors and vshift's own
  virtual mic are left out, since listening to those would make vshift hear itself.
- **If your system default mic is vshift's own `vshift_mic_input`** (PipeWire can make a
  newly created virtual mic the default), vshift now notices, listens to a real mic instead
  and says so; `vshift doctor` flags it too. Before, vshift would quietly listen to its own
  output. Set your real mic as default in your sound settings to fix it for other apps.
- How it works: many PortAudio builds (including Arch/CachyOS's) only see raw ALSA
  hardware, and opening a mic that way would lock it away from Discord. vshift records
  through PortAudio's `pulse` device and points it at the chosen mic with `PULSE_SOURCE`
  (or `pipewire` with `PIPEWIRE_NODE`), the same trick it already used for output. If
  neither device exists, install `pulseaudio-alsa` or `pipewire-alsa`.
- The full mic id is what gets saved, so it survives reboots; if a saved mic is unplugged,
  vshift warns and uses the default. `--input-device N` (a number from `vshift devices`)
  still works.
- The panel is one row taller again: it now needs 23 terminal rows for the full view.

### Voice browser (`V`)

`V` now opens a searchable list of every English voice vshift knows: the Piper
catalog, all 28 Kokoro voices, and anything you downloaded yourself.

- Type to filter. Every word must match, so `uk female`, `kokoro`, `amy` or `high` all
  work, and `male` does not match `female`.
- `↑`/`↓` (and PgUp/PgDn/Home/End) move, `Enter` switches to the highlighted voice,
  `Esc` closes. `●` is the voice in use, `✓` downloaded, `↓` not downloaded yet; picking a
  `↓` voice downloads it in the background (the browser tells you roughly how big).
- **`Tab` (or `→`) plays a short sample** of a downloaded voice at your current speed and
  volume. In `vshift discord` it plays on your own speakers/headphones only (through
  `paplay`, or `pw-play`), never into the virtual mic, so your call doesn't hear it. It
  uses a separate TTS worker, so the voice you're using doesn't change.
- The mic is paused while the browser is open, and a sample still playing is stopped
  when you close it, so the mic can't pick it up.
- Typing a valid voice id that isn't in the list offers it as a custom id, so any Piper
  voice is still reachable, as with the old prompt.

### Translation (`L`)

Speak in any of Whisper's ~100 languages and vshift says it in English.

```bash
vshift discord --translate                  # detect the language you speak
vshift discord --translate --source-lang de # or tell it (more reliable on short lines)
```

- `L` turns it on and off live; the choice and `--source-lang` are saved.
- English-only Whisper models (`base.en`, the default) cannot translate, so translation
  uses the multilingual twin of your `--stt-model`: `base.en` → `base`, `small.en` →
  `small`. The first time, that model downloads (`base` is ~145 MB) and the panel shows
  "Loading…"; anything you say meanwhile waits in the queue. If loading fails, vshift
  says why and goes back to the old model.
- The panel's new `Translate` row shows `auto → English` (with the last language Whisper
  heard) or `de → English`.
- Quality follows the model size: `base` is fine for everyday phrases, `--stt-model small`
  is noticeably better but slower. `large-v3-turbo` was not trained to translate; avoid it
  here. Output is English only (Whisper translates *into* English), which matches the
  English voices.
- Tested here with real models: Spanish speech ("Hola amigos, ¿qué tal? Hoy vamos a jugar
  una partida juntos.") came back as "Hello friends, how are you? Today we are going to
  play a game together.", with the language detected as `es`.
- The panel is one row taller: it now needs 22 terminal rows for the full view.
- The Whisper worker's request format changed to carry the task and language
  (`stt_worker.py` and `STTClient` must be updated together).

### Fixes

- **A click or cough no longer records 15 seconds of silence.** A noise burst shorter than
  real speech used to keep the recorder running until its 15 s limit, then send all of it
  to Whisper (slow, and an invitation to hallucinate). It is now dropped as soon as the
  silence after it is long enough.
- **Typing "ok" in `T` (or repeating it with `R`) is spoken again.** The filter for
  Whisper's "okay" hallucinations now only applies to what the mic heard.
- **Voice downloads can't leave a broken file behind.** Piper's downloader wrote straight
  to the final `.onnx`, so an interrupted download looked installed but never loaded.
  Downloads now land in a temporary folder first. For a voice that is already damaged,
  the error says so and `vshift download <id> --force` fetches it again.
- After a first Kokoro voice downloads mid-session, the Kokoro voices join `[` / `]`
  straight away instead of after a restart. `--voices-dir ~/…` is expanded everywhere.
- **Panel:** a spinner while Whisper/TTS are working, a mic meter that is green above the
  threshold and red near clipping (with the live level as a number), sliders for speed and
  volume, the voice shown as name plus engine (`lessac-medium  piper US`), coloured keys
  and frame, and the version in the title.

## 1.0.7: Kokoro voices (a second, higher-quality engine)

Piper stays the default. Kokoro-82M is available as an opt-in engine for noticeably more
natural speech, **at the cost of speed** (see the numbers below). Pick a Kokoro voice by
prefixing its name with `kokoro:`:

```bash
venv/bin/pip install 'kokoro-onnx>=0.6,<1'     # only if you updated in place (install.sh does it for you)
vshift discord --voice kokoro:af_heart          # first use downloads ~350 MB once
vshift tts-bench                                # measure Piper vs Kokoro on YOUR machine
```

- 28 English voices (`vshift voices` lists them): `af_`/`am_` American female/male,
  `bf_`/`bm_` British. Once the model is downloaded they also join the `[` / `]` cycle,
  and the choice is saved like any other voice. Switching between Kokoro voices is
  instant (no reload).
- Downloads are checked against a fixed size and SHA-256 before use, so a cut-off
  download is rejected instead of being loaded (which would sound like static).
- Speed and volume keys work as before. Kokoro speaks at 24 kHz.
- `VSHIFT_KOKORO_MODEL=int8` selects the smaller 92 MB model (can be faster on CPUs with
  fast int8 maths; it was *not* faster on the one machine I measured). `VSHIFT_TTS_THREADS=N`
  sets how many CPU threads Kokoro uses (default: up to 4, since speech-to-text is
  running at the same time).

### The speed trade-off, measured

On the same 1-core 2.1 GHz virtual machine (a deliberately weak CPU; a desktop should be
much faster, so run `vshift tts-bench` yourself):

| Engine | Real-time factor | Typical 3 s sentence |
| --- | --- | --- |
| Piper (an older release of the lessac-medium voice) | 0.07 | about 0.2 s |
| Kokoro-82M | 0.8 | about 2.4 s |

Real-time factor is seconds to synthesize divided by seconds of speech. On that machine
Kokoro keeps up with speech but is roughly 10x slower than Piper, which adds a noticeable
delay to every reply. If you want the lowest delay, stay on Piper; if you want the best
voice and your CPU is fast, try Kokoro.

### Why Kokoro (and why not the others)

I compared what runs usefully on a CPU for a live, sentence-at-a-time voice changer.
Quality and speed figures below are from third-party benchmarks on a shared CPU, so treat
them as relative, not absolute (UTMOS is an automatic naturalness score, higher is better).

| Model | Verdict for vshift |
| --- | --- |
| **Kokoro-82M** | Chosen. Best measured quality of the CPU-friendly options (UTMOS about 4.44), Apache-2.0, plain ONNX Runtime (no PyTorch), 28 English voices, works with numpy 2. No voice cloning. |
| Pocket TTS (Kyutai) | Good, and it clones voices from a short sample, but quality scored lower (about 4.10) and it needs PyTorch (a very large install on Linux unless you use the CPU wheel index). Best candidate to add later if you want cloning. |
| Supertonic 3 | Fastest decent option (about 4x real-time at 5 steps, UTMOS about 4.3). Strong runner-up, but I could not test it here, and the sources I read disagree on its license (OpenRAIL-M in one benchmark, MIT in another), so check that before relying on it. |
| Chatterbox-Turbo | Aimed at GPUs; one guide reports 10 to 30 s per generation on CPU. Turbo needs a reference clip, one guide says it needed an older numpy than vshift uses, and its output carries an embedded audio watermark. (Its smaller Chatterbox-Nano is claimed to run about 3x real-time on 8 cores; I did not evaluate it.) |
| Dia | GPU only (about 10 GB VRAM). A dialogue model, not built for sentence-by-sentence use. |
| VibeVoice-Realtime | Its published latency figures assume a GPU, and its model card limits it to research use and lists real-time voice conversion for calls as out of scope, which is what this tool does. |

Kokoro was also the only one I could run end to end here, against the real model files.

## 1.0.6: volume

- **Voice volume.** `.` (or `>`) louder, `,` (or `<`) quieter, in 10% steps from 10%
  to 200%. It is applied to the speech after synthesis, so it works with every voice,
  takes effect from the next sentence, and is saved between runs. It also changes
  what you hear when monitoring is on, since monitoring plays the same audio.
- Boosting clips at the loudest level the audio format allows rather than wrapping
  around, but above about 150% a loud voice can sound crunchy, and vshift says so.
  For more volume than that, raise your Discord input volume or the system mixer.
- The panel is one row taller (it now needs 21 terminal rows for the full view; below
  that you get the compact view).

## 1.0.5: playback is per-clip again by default

1.0.2 added a persistent output stream to avoid tiny gaps between sentences.
On one real PipeWire setup (CachyOS) it made speech sound like static, while the
old open-a-stream-per-sentence playback and a Piper-generated WAV were both clean.
So the old playback is the default again. The persistent stream is still there as an
opt-in experiment, and I don't know why it misbehaves:

```bash
VSHIFT_PERSISTENT_PLAYBACK=1 vshift discord                      # persistent stream
VSHIFT_PERSISTENT_PLAYBACK=1 VSHIFT_LATENCY=high vshift discord  # ...with a bigger buffer
```

`VSHIFT_LATENCY` accepts `low`, `high` or a number of seconds (e.g. `0.3`) and only
applies to the persistent stream. Note that `vshift test-mic` always uses the
per-clip playback, so it does not exercise the persistent stream.

The 150 ms echo pause fix from 1.0.3 applies to both playback modes: with monitoring
off there is no added pause between sentences.

## 1.0.3

- **Speed keys.** `-` slower, `+` (or `=`, no Shift needed) faster, `0` back to the
  voice's own default. The panel shows speed as a rate (`1.18x`), so `+` always
  makes it larger. Steps are 0.05 and range from 0.5x to 2.0x. Pressing a key from
  "voice default" starts from normal speed (length_scale 1.0). Saved between runs.
- **Auto-calibrate threshold (`C`).** Stay quiet for 2 seconds; vshift measures
  your room noise and sets the threshold above it (rounded to the same 25-step
  grid as `↑`/`↓`). While it runs, nothing you say is sent to speech-to-text. If you
  talk during the measurement it refuses ("Too loud to calibrate") and keeps your old
  threshold. Pausing cancels it.
- **Repeat (`R`)** speaks the last line again, whether it was heard or typed. It
  doesn't add to the utterance count.
- **Typed-text history.** In the `T` prompt, `↑`/`↓` walk through lines you typed
  earlier this session (what you were typing is kept, so `↓` brings it back). History
  is not saved to disk.
- **Correction to 1.0.2's "gapless playback".** 1.0.2 kept the stream open, but
  `playback_stage` still slept 150 ms after every clip, so there was still a silent
  gap between sentences. That pause exists to let echo die down, so it now only runs
  when the voice can reach your mic (monitoring on, or `vshift run` through
  speakers). Plain `vshift discord` with monitoring off has no gap at all.

## 1.0.2 polish

- **Persistent output stream (opt-in since 1.0.5; see above, it caused static on one
  setup).** Playback used `sd.play`, which opens and closes an output
  stream for every sentence. This keeps one output stream open and writes
  each sentence into it, reopening only if the sample rate changes or a write
  fails. The stream is closed after 5 idle seconds, so vshift doesn't hold your
  output device open while you're silent.
- **Mic level meter.** A new `Mic level` row shows your live input level with a
  `┃` marker at the threshold. Bars past the marker mean that audio counts as
  speech, so you can set `↑`/`↓` by eye instead of guessing. The panel is one
  row taller (at that point it needed 20 terminal rows for the full view; below that you get
  the compact view).
- **Pausing no longer corrupts the next utterance.** Pausing mid-sentence used
  to leave the half-captured speech in the buffer, and on resume it was glued
  to the silence that followed. Pausing now discards it.
- **Virtual sink not visible to PortAudio.** `vshift` finds its virtual sink by
  name in PortAudio's device list. A PortAudio built without a PulseAudio/PipeWire
  host API only lists ALSA devices, so the sink never shows up and startup used
  to fail. vshift now falls back to the ALSA `pulse` device and steers it to the
  sink with `PULSE_SINK`. If neither works, the error says what to try
  (on Arch/CachyOS, `pulseaudio-alsa` may add the `pulse` device).
- **`install.sh` on CachyOS/Arch.** The system Python there is newer than 3.13, so
  the installer now explains how to get one (`yay -S python313`, or `uv python
  install 3.13`) and accepts `VSHIFT_PYTHON=/path/to/python ./install.sh`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Set `VSHIFT_TEST_KOKORO_DIR` to a folder holding the downloaded Kokoro files and run
`tests.test_kokoro_real` with the vshift venv's Python to also test against the real model.

The other tests use a fake `sounddevice` and a fake `pactl`, so they need no audio
hardware, PipeWire, Whisper or Piper. They cover the persistent output stream,
pause handling, the meter and panel layout, sink routing, speed keys, calibration,
repeat and history, and one end-to-end run of the real curses UI in a
pseudo-terminal (`tests/test_tui_pty.py`, takes about 5 seconds). They do **not**
prove anything about real audio devices.

## 1.0.1 fixes

- **"Hear generated" (`H`) now works and tells you why when it can't.** The
  loopback used to fail silently while the UI still said "ON". Now the toggle
  reflects what actually happened, shows the real error on screen, and logs to
  `~/.cache/vshift/monitor.log`. Loading falls back through simpler
  `module-loopback` argument sets (PulseAudio and pipewire-pulse accept
  different options), never loops the virtual sink into itself, and finds a
  real output even if your default sink was switched. `vshift doctor` checks
  the default sink too. In `vshift run` (audio already plays on your speakers)
  the toggle is shown as not applicable instead of doing nothing.
- **Cleaner, always-centered terminal UI.** The whole interface is one panel
  that is re-centered horizontally *and* vertically from the live terminal size
  every frame, so it stays in the middle when you resize, maximise or split the
  window. It has a constant height (no jumping when messages appear), aligned
  label/value columns, key hints next to the settings they control, coloured
  status, wrapped transcripts and errors (nothing cut off), and a compact centered
  view for very small terminals. `T` (type to speak) and `V` (voice id) open
  centered input boxes over the live panel; long text scrolls instead of being
  cut off, `Esc` cancels instantly, and the status keeps updating underneath.
- **No more crash when PortAudio is missing.** `--help`, `--version`, `config`,
  `voices` and `doctor` now work without it, and audio commands print the
  install command for your distro.
- **Echo guard no longer eats real speech** (see above).
- **First launch downloads the Whisper model without timing out** (was a fixed
  60 s); real worker errors are shown instead of a generic timeout, and
  half-started workers are cleaned up. The TTS startup wait is also bounded.
- **Errors are tracked per source** so one stage succeeding no longer hides
  another stage's error; audio-output failures now show in the UI instead of
  garbling the terminal.
- Whisper loop-collapse is more conservative ("no no no no" and laughter are
  kept). `--threshold` help now shows the real default (400). `test-mic` always
  cleans up its virtual devices. `speaker_id: true` in settings is rejected.
  Exact-match checks when removing virtual devices. `install.sh --system` also
  supports apt and dnf.

If "Hear generated" still fails, the on-screen message and
`~/.cache/vshift/monitor.log` contain the exact `pactl` error, and
`pactl list short modules | grep loopback` shows what is loaded.

## 1.0.0 fixes

The TTS worker now passes only explicitly configured Piper synthesis options. This avoids overriding Piper's numeric volume default with `None`, which caused `unsupported operand type(s)` during playback. TTS worker exceptions also include their Python exception type and a full traceback in `~/.cache/vshift/tts_worker.log`.

## Fixed: repeated/looping responses

Users could hit vshift repeating the same line over and over. Two independent causes, both addressed:

1. **Acoustic feedback.** With local monitoring on (previously the default) and real speakers near an active mic, the AI's own voice could get picked back up, transcribed, and re-spoken indefinitely. Monitoring now defaults to off, and regardless of whether it's on, vshift discards anything heard that closely matches what it just said within the last few seconds (see above).
2. **Whisper decode loops.** On ambiguous/noisy short clips, faster-whisper could occasionally decode the same short phrase repeatedly within a single transcription (a known degenerate-decode failure mode, unrelated to feedback). The Whisper worker now also filters on `compression_ratio` (a standard signal for this exact failure mode) and collapses any transcript that's dominated by a short repeating phrase down to one occurrence before it's spoken.

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

