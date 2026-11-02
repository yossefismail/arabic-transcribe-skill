# Arabic Transcribe

**English** · [العربية](README.ar.md)

An agent skill that turns an Arabic voice note, meeting recording or interview
into a **single self-contained HTML page**: embedded audio player, speaker-coloured
transcript that highlights as it plays, click-to-seek text, a written summary and
an action-points checklist — dressed in a "technical drafting sheet" layout with an automatically generated
key-moments rail on the timeline.

Built for 19 Arabic dialects with code-switched English, French, Spanish, or Italian. (WhatsApp voice notes,
voice memos, calls).

**Everything runs on your machine.** No cloud, no upload, no account, no token —
the audio, the models and the finished player all stay on disk. The only network
traffic is the one-time model download.

```sh
npx skills add yossefismail/arabic-transcribe-skill
```

![Arabic Transcribe workflow: eight stages from a voice note to a drafting-sheet player](assets/workflow.png?v=2)

## What you get

| Output | What it is |
|---|---|
| `<name> - arabic transcript.html` | The deliverable. Audio embedded as base64, so it works offline and can be sent as one file. Speaker colours, live clause highlight, click-to-seek, summary card with checkboxes that remember their state. |
| `<name> - arabic transcript.txt` | Plain-text fallback with RLM/LRM bidi marks so Arabic + English mix correctly in Notepad. |
| `<name> - enhanced audio.m4a` | The cleaned-up audio on its own. |

## Model and accuracy

Transcription is [`IFM/Jais-ASR`](https://huggingface.co/IFM/Jais-ASR) (5.72 B
params, Apache-2.0, loaded in 4-bit). Published error rates on the [Open Universal
Arabic ASR Leaderboard](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard)
(lower is better; macro-average of six sets):

| Test set | WER % | CER % |
|---|---:|---:|
| **Average** | **22.85** | **8.37** |
| Common Voice 18 — MSA read speech | 8.56 | 2.70 |
| MGB-2 — MSA broadcast news | 9.59 | 3.92 |
| MASC clean — MSA lectures | 16.12 | 3.76 |
| MASC noisy — MSA lectures, noisy | 25.44 | 7.51 |
| SADA — Saudi dialect, broadcast | 30.85 | 14.33 |
| Casablanca — 8 dialects, conversational | 46.56 | 18.02 |

That average beats the other models on the same leaderboard (Audar 23.17,
Speechmatics Melia 1 23.51, ElevenLabs Scribe v2 24.54, Cohere 25.87), though not
every individual set. Conversational multi-dialect audio is the hard case — plan
for roughly 5× the error of clean read speech, and note these are bf16 numbers
while the skill runs 4-bit.

**Coverage:** Modern Standard Arabic, 19 dialects (`DZ` Algerian, `BH` Bahraini,
`EG` Egyptian, `AE` Emirati, `IQ` Iraqi, `JO` Jordanian, `KW` Kuwaiti, `LB`
Lebanese, `LY` Libyan, `MR` Mauritanian, `MA` Moroccan, `OM` Omani, `PS`
Palestinian, `QA` Qatari, `SA` Saudi, `SD` Sudanese, `SY` Syrian, `TN` Tunisian,
`YE` Yemeni), plus code-switching into English, French, Spanish and Italian — the
foreign spans stay in Latin script. `--language auto` lets the model detect the
dialect; `--language EG` (the default) pins it.

Full detail, protocol and citation: [references/model.md](references/model.md).

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

One command, via the [skills CLI](https://github.com/vercel-labs/skills):

```sh
npx skills add yossefismail/arabic-transcribe-skill
```

It detects your agents and installs the skill into the right place — project-level
`.agents/skills/arabic-transcribe`, or with `-g` the agent's user directory
(`~/.config/opencode/skills/arabic-transcribe` for OpenCode,
`~/.claude/skills/arabic-transcribe` for Claude Code). Project installs are
recorded in `skills-lock.json` with the source and a content hash.

```sh
npx skills add yossefismail/arabic-transcribe-skill -g -a opencode     # user-level, OpenCode
npx skills add yossefismail/arabic-transcribe-skill -g -a claude-code  # user-level, Claude Code
npx skills list                                                        # what's installed
npx skills update                                                      # pull newer commits
npx skills remove arabic-transcribe                                    # uninstall
```

This is a single-skill repo, so no `--skill` flag is needed.

Prefer a plain copy? Clone it where your agent looks for skills:

```sh
git clone https://github.com/yossefismail/arabic-transcribe-skill.git \
  ~/.config/opencode/skills/arabic-transcribe        # OpenCode
# or ~/.claude/skills/arabic-transcribe             # Claude Code
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

Useful flags: `--quant 8bit`, `--no-enhance`, `--max-seconds N`,
`--language EG|auto|<dialect>`.

## Fully local

- **Everything runs locally.** Audio never leaves the machine. There is no
  telemetry and no upload step anywhere in the scripts.
- Network traffic is limited to first-run downloads: the Jais-ASR weights from the
  Hugging Face Hub, and the NVIDIA diarizer (~8 MB `nemo-speech` runtime + 102 MB
  GGUF) from NVIDIA's GitHub release and model host. **No Hugging Face account,
  token, or gated-model signup is involved anywhere.**
- The player is a single file with the audio embedded as base64 and the fonts
  inlined, so it opens on a machine that has never seen this repo and has no
  network at all.
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
references/model.md       Jais-ASR error rates, dialect codes, code-switching
references/ui-system.md   the visual system, tokens and layout rules
assets/fonts/             Archivo, JetBrains Mono, Cairo (OFL, see OFL.txt)
assets/workflow.png       workflow diagram, English
assets/workflow-ar.png    workflow diagram, Arabic (RTL)
README.ar.md              النسخة العربية من هذا الملف
```

## Licence

Code and documentation: MIT (see [LICENSE](LICENSE)).
Bundled fonts: SIL Open Font License 1.1 (see
[assets/fonts/OFL.txt](assets/fonts/OFL.txt)).
Jais-ASR weights: Apache-2.0. `nemo-speech` runtime: Apache-2.0.
Nemotron-3-Diarization weights: OpenMDW-1.1.
