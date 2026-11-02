# Arabic Transcribe

An agent skill that turns an Arabic voice note, meeting recording or interview
into a **single self-contained HTML page**: embedded audio player, speaker-coloured
transcript that highlights as it plays, click-to-seek text, a written summary and
an action-points checklist — dressed in a "technical drafting sheet" layout with a
key-moments rail on the timeline.

Built for Egyptian-dialect Arabic with code-switched English (WhatsApp voice notes,
voice memos, calls), but it handles MSA and other dialects too.

```
audio ─► denoise + speech EQ ─► NVIDIA Sortformer diarization ─► Jais-ASR (4-bit)
      ─► agent writes summary ─► drafting-sheet HTML + plain-text transcript
```

## What you get

| Output | What it is |
|---|---|
| `<name> - arabic transcript.html` | The deliverable. Audio embedded as base64, so it works offline and can be sent as one file. Speaker colours, live clause highlight, click-to-seek, summary card with checkboxes that remember their state. |
| `<name> - arabic transcript.txt` | Plain-text fallback with RLM/LRM bidi marks so Arabic + English mix correctly in Notepad. |
| `<name> - enhanced audio.m4a` | The cleaned-up audio on its own. |

## Requirements

- **Python 3.12** (`py -3.12` on Windows, `python3.12` elsewhere)
- **NVIDIA GPU with CUDA**, 6 GB VRAM or more (tested on an RTX 4050 6 GB). The
  device-map budget is sized from free VRAM at load time; `JAIS_GPU_MEM` and
  `JAIS_CPU_MEM` override it.
- ~12 GB free disk for the quantized model, plus the HF cache
- **No Hugging Face account, token, or gated-model signup.** Diarization runs on
  NVIDIA Nemotron-3-Diarization through NVIDIA's `nemo-speech` runtime, which
  installs itself on first use (~110 MB total, SHA-256 verified).

Windows is the tested platform. macOS/Linux should work — the only Windows-specific
paths in the code are guarded — but they are untested.

## Install

The skill is a folder. Copy it where your agent looks for skills:

```sh
# OpenCode
cp -r arabic-transcribe ~/.config/opencode/skills/

# Claude Code
cp -r arabic-transcribe ~/.claude/skills/
```

On Windows the OpenCode path is `C:\Users\<you>\.config\opencode\skills\`.

Then install the Python dependencies once (see
[references/setup.md](references/setup.md) for the full walkthrough, including the
curl salvage loop if the 11.4 GB model download stalls):

```sh
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
py -3.12 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cu128
py -3.12 -m pip install transformers==5.16.1 bitsandbytes soundfile imageio-ffmpeg huggingface-hub noisereduce
py -3.12 -m pip install --no-deps resemble-enhance tabulate
```

Nothing else: the diarizer and its runtime are fetched automatically on first use.

Verify: `py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
must print a `+cu128` build and `True`.

## Use

Ask your agent to transcribe a recording, or drive the three scripts yourself:

```sh
# 1. transcribe + diarize + build the player
py -3.12 scripts/run_pipeline.py "voice note.m4a" --outdir out

# 2. write the summary (agent step) and inject it into both files
py -3.12 scripts/add_summary.py "out/voice note" --summary-file summary.md

# 3. apply the house UI and key-moment ticks
py -3.12 scripts/apply_ui.py "out/voice note" \
  --title "تفريغ|المكالمة" --project "MEETING NOTES" \
  --date "01·11·2026" --kicker "RECORDING 04 — TRANSCRIPT INDEX"
```

`SKILL.md` is the full operating manual the agent follows: how to run the
pipeline, what a good summary contains, and the house UI rules.

Useful flags: `--quant 8bit`, `--no-enhance`, `--max-seconds N`.

## Security and privacy

- **Everything runs locally.** Audio never leaves the machine. There is no
  telemetry and no upload step anywhere in the scripts.
- Network traffic is limited to first-run downloads: the Jais-ASR weights from the
  Hugging Face Hub, and the NVIDIA diarizer (~8 MB `nemo-speech` runtime + 102 MB
  GGUF) from NVIDIA's GitHub release and model host. **No Hugging Face account,
  token, or gated-model signup is involved anywhere.**
- The `nemo-speech` binary is downloaded and **executed**, so the version is pinned
  (v0.3.0) and checked against the release's published SHA-256 before it runs. The
  runtime is Apache-2.0; the model is OpenMDW-1.1. Delete
  `~/.cache/arabic-transcribe/nemo-speech/` to remove it.
- `IFM/Jais-ASR` ships custom modelling code that `transformers` executes
  (`trust_remote_code=True`), so the pipeline **pins an audited commit**
  (`JAIS_REVISION`); set `JAIS_REVISION=main` only if you want to track the hub.
- Nothing is installed into your Python environment: the `deepspeed` inference
  stub that Resemble Enhance needs is written to a temp dir and added to
  `sys.path`.
- The generated HTML embeds the **full audio**. Treat transcripts as recordings,
  not as text: sharing the file shares the audio.

## Troubleshooting

See [references/setup.md](references/setup.md) — it covers the model download
that stalls, `paging file too small`, the bitsandbytes device-map error, and
`nemo-speech` diarization problems.

## Layout

```
SKILL.md                  agent instructions (workflow, summary spec, UI rules)
scripts/run_pipeline.py   decode → enhance → diarize → transcribe → HTML/TXT
scripts/add_summary.py    inject the agent-written summary + checklist
scripts/apply_ui.py       drafting-sheet theme, player, key-moments rail
references/setup.md       environment + model setup, error table
references/ui-system.md   the visual system, tokens and layout rules
assets/fonts/             Archivo, JetBrains Mono, Cairo (OFL, see OFL.txt)
```

## Licence

Code and documentation: MIT (see [LICENSE](LICENSE)).
Bundled fonts: SIL Open Font License 1.1 (see
[assets/fonts/OFL.txt](assets/fonts/OFL.txt)).
Jais-ASR weights: Apache-2.0. `nemo-speech` runtime: Apache-2.0.
Nemotron-3-Diarization weights: OpenMDW-1.1.
