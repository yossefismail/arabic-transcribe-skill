---
name: Arabic Transcribe
description: Transcribe Arabic voice notes (Egyptian dialect, mixed Arabic/English) with Jais-ASR, diarize speakers with NVIDIA Nemotron-3-Diarization (no Hugging Face account or gated-model signup), and build a synced HTML player with embedded audio and speaker-colored text, ending with a summary and an action-points checklist. Use when the user asks to transcribe, summarize, or extract action items from a WhatsApp voice note, voice memo, or Arabic audio recording with speaker labels.
---

# Arabic Transcribe

Transcribes Arabic (Egyptian dialect, code-switched English) audio using **Jais-ASR in 4-bit NF4** (the recommended default), diarizes speakers with **NVIDIA Nemotron-3-Diarization** (Sortformer V3, up to 8 speakers, no Hugging Face account or gated signup), and produces a **single self-contained HTML file**: embedded audio player + transcript where the current clause highlights during playback, clicking text seeks the audio, and speakers are color-coded. The transcript **ends with a summary of the recording and an action-points checklist** (interactive checkboxes in the HTML) — written by you after transcription and injected with `scripts/add_summary.py`. The player is then dressed in the skill's **"drafting sheet" UI** — paper grid, hard ink rules, mono technical labels, an oversized Arabic headline, and a **key-moments rail on the recording timeline** — applied with `scripts/apply_ui.py` (see `references/ui-system.md`).

## Pipeline entry point

Everything runs through one script — `scripts/run_pipeline.py`:

```sh
py -3.12 "<base>/scripts/run_pipeline.py" "<input audio>" --outdir "<output dir>"
```

`py -3.12` is the Windows launcher — on macOS/Linux use `python3.12`. The scripts
are otherwise platform-neutral; the Windows-specific quirks below are handled
only when they apply.

- Works on mp4/m4a/mp3/wav/ogg/opus (anything ffmpeg reads).
- **Enhances the audio before ASR** (and for the player): neural denoise first — Resemble Enhance (16 kHz in → full-band 44.1 kHz out, GPU, ~0.1× realtime) — then 90 Hz high-pass, speech-band EQ (mud cut 250 Hz, vocal presence boost 2.5 kHz, consonant lift 5 kHz), chunked compressor/AGC (+10 dB max lift), loudness normalized to −14 dBFS. Measured on a test recording: +6 dB overall, +7.5 dB on speech frames. `--no-enhance` to skip. This stage also exists as a standalone skill: `audio-enhance`. If Resemble Enhance is missing/broken the script falls back to spectral `noisereduce` automatically.
- **No separate VAD step**: the full file is diarized first and the diarizer's speaker turns double as the speech map. Blocks are merged (gaps ≤ 15 s), blocks with < 3 s of talk are dropped, and only those blocks are transcribed — long noise/TV-chatter tails are skipped automatically. Faint background speech that the diarizer detects but ASR can't transcribe comes out as "(no speech detected)" via Jais's refusal mechanism.
- Outputs next to the input (or `--outdir`):
  - `<name> - arabic transcript.html` — the synced player (main deliverable; preview it to the user with the browser preview tool; the pipeline emits a plain version — `scripts/apply_ui.py` applies the house UI)
  - `<name> - arabic transcript.txt` — plain-text fallback with bidi marks (RLM/LRM) so Arabic+English renders correctly in Notepad
  - the pipeline does **not** write the summary — you write it after transcription (Workflow step 4) and inject it into both files with `scripts/add_summary.py`
- **Model**: `IFM/Jais-ASR` (5.72 B params, Apache-2.0) — MSA, 19 Arabic dialects, and code-switching into English, French, Spanish and Italian. Published WER/CER, the dialect codes, and the `--language` pin are in `references/model.md`.
- Optional: `--quant 8bit` (default is 4bit NF4), `--max-seconds N` to cap processing, `--language EG|auto|<dialect>` to pin the dialect (default `EG`; `auto` lets Jais detect it).

## Workflow

1. **Check the environment** first:
   ```sh
   py -3.12 -c "import torch, transformers, bitsandbytes, soundfile, imageio_ffmpeg, resemble_enhance, tabulate; print(torch.cuda.is_available())"
   ```
   If anything is missing or CUDA is False, follow `references/setup.md` (torch must be the cu128 build; do NOT let pip replace it with a CPU wheel).
2. **No Hugging Face account, token, or gated-model signup is needed anywhere.** Speaker diarization uses NVIDIA Nemotron-3-Diarization through NVIDIA's `nemo-speech` CLI, which the pipeline downloads automatically (pinned release, SHA-256 verified: ~8 MB runtime + 102 MB model, both from GitHub/NVIDIA, no login). Never ask the user for an HF token.
3. **Run the pipeline** in the background (4-bit model load takes ~1 min; transcription ~1-3 min per minute of audio on a 6 GB GPU).
4. **Summarize the recording** (after the pipeline finishes) — read the transcript TXT, then write a summary file (UTF-8) and inject it:
   ```sh
   py -3.12 "<base>/scripts/add_summary.py" "<input path without extension>" --summary-file "<summary.md>"
   ```
   The summary is markdown-lite. **Be comprehensive**: cover the whole recording end-to-end — every topic, decision, and stretch of talk, including the last segments (don't trail off after the first windows). Scale depth to the material: a 2-minute note gets a short paragraph, a long call gets a detailed, sectioned summary.
   **Don't be formulaic**: the parts below are a guide, not a template to fill mechanically — adapt wording, structure, and length to what the recording actually contains, and never pad with boilerplate.
   **Transcripts contain ASR errors**: mis-heard names, wrong numbers, garbled code-switched words, occasionally invented phrases. Read critically — prefer the plausible reading of garbled words, cross-check details that repeat, hedge or mark unclear names/numbers instead of confidently repeating a dubious transcription, and never invent facts the transcript doesn't support.
   Typical parts (adapt freely):
   - **Overview** — a `**نظرة عامة:**` paragraph: what the recording is, who speaks, what it's about.
   - **Key points** — a `**أهم النقاط:**` heading + `- ` bullets of the main things said, with minute marks.
   - **Action points** — a `**بنود التنفيذ / Action points:**` heading + checklist items as task lines: `- [ ] task` (or `- [x]` if the recording says it was already done). There is **no fixed count** — extract *every* distinct commitment, request, and "we need to...", however many the recording contains (two or twenty); include who is responsible and the deadline when stated. If there are truly none, state that in one line instead of inventing tasks.
   Write in MSA for Arabic recordings, keeping names and code-switched English terms in Latin script. The script appends the summary to the end of both the HTML (styled card; `- [ ]` items become checkboxes whose state persists in localStorage) and the TXT (`☐`/`☑`). It is idempotent — re-running after editing the summary file replaces the previous summary.
5. **Apply the house UI + key moments** (after the summary, so the moments can be derived from it):
   ```sh
   py -3.12 "<base>/scripts/apply_ui.py" "<input path without extension>" \
     --title "تفريغ|المكالمة" \
     --kicker "RECORDING 04 — TRANSCRIPT INDEX" \
     --note "..." --project "MEETING NOTES" --date "01·11·2026" \
     [--moments "<moments.json>"] [--end "164:14"]
   ```
   (`py -3.12` is the Windows launcher; elsewhere use `python3.12`.)
   This replaces the pipeline stylesheet with the drafting-sheet system, rebuilds the title block/hero/sticky player, and puts **key-moment ticks on the recording rail**. If `--moments` is omitted it derives them from the summary's `(mm:ss–mm:ss)` headings; pass a JSON list (`[{"t": "44:31", "label": "هامش الربح"}]`) for shorter, better labels. Idempotent: re-run it after editing the summary or the moments. Full spec: `references/ui-system.md`.
6. **Preview the HTML** to the user (browser preview tool) and report: detected speakers (A/B/C with who dominates which stretch), speech region used, the caveat that within 30 s windows the highlight timing is proportional, not word-exact, and a recap of the action points.

## UI system ("drafting sheet")

The player has one house look, defined in `references/ui-system.md` and generated
by `scripts/apply_ui.py` (fonts in `assets/fonts/`, inlined as base64 so the file
stays offline). It borrows the visual language of
[final-final-v1.com](https://final-final-v1.com/): warm grid paper, hard ink
rules, mono technical labels, a single vermillion accent, an oversized display
headline, and numbered sections.

Layout: title block (Project / Date / Speech / Clauses) → hero (eyebrow, Arabic
headline, lede, fact list) → **sticky player with the key-moments rail** →
summary sheet → transcript index (mono time gutter + text, active window marked)
→ colophon.

The rail is a custom timeline (`role="slider"`), not the native control: every
key moment is a numbered tick, the played part turns accent, the `nowline` names
the current moment, and clicking a tick seeks and plays. `space` toggles playback,
`←/→` ±5 s, `Shift+←/→` ±30 s, `PageUp/Down` ±60 s, `Home/End` jump; the `1×`
button cycles speed. Don't put the native `<audio controls>` back next to it.

## Known pitfalls (all pre-solved in the scripts — do not improvise)

- **Jais must load with `llm_int8_enable_fp32_cpu_offload=True`** or bitsandbytes rejects the device map on ≤6 GB GPUs ("Some modules are dispatched on the CPU").
- **Diarization runs out-of-process through NVIDIA's `nemo-speech` CLI** (native C++, Apache-2.0), not through a Python package. First use downloads the pinned CPU build (~8 MB) from the NeMo-Speech.cpp GitHub release, verifies its published SHA-256, and unpacks it into `~/.cache/arabic-transcribe/nemo-speech/<version>/`. The CLI then pulls the 102 MB Nemotron-3-Diarization GGUF into its own cache (`%LOCALAPPDATA%\NeMoSpeech\models` on Windows) and verifies that too. `NEMO_SPEECH_BIN` points at another build (e.g. the Windows CUDA zip), `NEMO_DIAR_DEVICE=cpu|cuda`, `NEMO_DIAR_PRESET` overrides the streaming preset.
- **`--preset v3-offline` is a large-chunk streaming pass**, roughly 10× faster than the default on short files and unlimited in duration; the pipeline retries with the default pass if a build rejects the preset. Sortformer derives the speaker count itself (up to 8) — there is no way to force an exact count.
- **First Jais download is ~11.4 GB.** On this network huggingface_hub (Xet backend) stalls silently — if the model download hangs, use the curl resume loop in `references/setup.md` instead.
- **Windows error 1455 ("paging file too small")** during model load = commit limit, not disk. Close apps / retry alone; don't add a pagefile.
- **ffmpeg is not on PATH** — use `imageio_ffmpeg.get_ffmpeg_exe()`.
- **Resemble Enhance on Windows** (all handled inside `run_pipeline.py`): its checkpoint YAMLs contain `PosixPath` tags → on Windows the script aliases `pathlib.PosixPath = pathlib.WindowsPath` before import (POSIX systems load them natively); it imports the training-only `deepspeed` package (unbuildable on Windows/py3.12) → the script writes a minimal inference-only stub to a **temp dir** and puts it on `sys.path` (your Python environment is not modified); it also needs `tabulate` at import time.
- Mixed Arabic/Latin text in plain files needs **RLM (U+200F) at the start of Arabic lines and LRM (U+200E) on LTR label lines**, and UTF-8 with BOM for Notepad.
- In HTML, mixed Arabic/English paragraphs must use `dir="auto"`; never hardcode `dir="rtl"` on a paragraph that starts with Latin text.
- **Never hand-edit the summary into the HTML with find/replace** — always go through `add_summary.py`; it owns the CSS, the checkbox persistence script, and idempotent replacement (marker `id="recording-summary"` in HTML, `SUMMARY (الملخص)` in TXT).
- **Never hand-edit the theme CSS/markup either** — `apply_ui.py` replaces the whole `<style id="ui-theme">` block, the `<!-- ui:chrome -->` prologue and `<script id="ui-player-js">` on every run. Change the script and re-run.
- Run `add_summary.py` **before** `apply_ui.py`. Re-running the summary afterwards restores plain paragraphs (the theme still works); re-run `apply_ui.py` after it to refresh the rail from the new summary headings.
- **Arabic must never get `letter-spacing`** (it breaks the joins) and shouldn't drop below 16 px; tracking belongs on the Latin/mono labels only.
- Key-moment times can exceed 99 minutes (`164:14`) — timestamp parsing must allow 1–3 minute digits, never assume `mm:ss` is 2 digits.
- `apply_ui.py` embeds three OFL webfonts (~190 KB base64) so the file stays self-contained. Pass `--no-fonts` only when system fonts are acceptable.
- If the speech end can't be detected (no `.sub` line and no timestamp range in the summary), pass `--end mm:ss`; otherwise the rail's moment positions fall back to the audio duration once metadata loads, but the header reads `00:00`.

## Requirements

- Python 3.12 (`py -3.12` on Windows, `python3.12` elsewhere), an NVIDIA CUDA GPU (tested: RTX 4050 6 GB; the device-map budget is sized from free VRAM at load time, override with `JAIS_GPU_MEM`/`JAIS_CPU_MEM`), torch cu128, transformers 5.16.1, bitsandbytes, torchaudio cu128, soundfile, imageio-ffmpeg, huggingface-hub, resemble-enhance (+ tabulate), noisereduce (fallback). Diarization needs no Python package — the `nemo-speech` binary installs itself.
- First run downloads Jais-ASR (~11.4 GB, Apache-2.0, ungated) + the NVIDIA diarizer (~110 MB: 8 MB `nemo-speech` CPU build + 102 MB Nemotron-3-Diarization GGUF, both SHA-256 verified). `IFM/Jais-ASR` ships custom modelling code run through `trust_remote_code=True`, so the pipeline pins an audited commit (`JAIS_REVISION`); set `JAIS_REVISION=main` to track the hub instead.
