# Model — Jais-ASR (IFM)

The transcription model is [`IFM/Jais-ASR`](https://huggingface.co/IFM/Jais-ASR),
released by IFM in October 2026. It is an Arabic speech-recognition LLM built on
Jais-3-4B: 5.72 B parameters, 11.4 GB in bf16, Apache-2.0. The skill loads a
pinned commit (`JAIS_REVISION`) in 4-bit NF4.

Everything below is taken from the model card and the benchmark it links to, not
from our own measurements. Two things are worth separating: the card's published
error rates, and what this skill actually does with the model (a 4-bit
quantization of it, on short windows).

## Error rates (as published)

Protocol: [Open Universal Arabic ASR
Leaderboard](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard),
six public test sets, evaluated with the leaderboard's own script. Lower is
better. `Average` is the macro-average over the six sets.

| Test set | What it covers | WER % | CER % |
|---|---|---:|---:|
| **Average** | six sets, macro | **22.85** | **8.37** |
| Common Voice 18 | MSA read speech | 8.56 | 2.70 |
| MGB-2 | MSA broadcast news | 9.59 | 3.92 |
| MASC clean | MSA lectures, clean | 16.12 | 3.76 |
| MASC noisy | MSA lectures, noisy | 25.44 | 7.51 |
| SADA | Saudi dialect, broadcast | 30.85 | 14.33 |
| Casablanca | 8 dialects, conversational | 46.56 | 18.02 |

Average WER of the models compared on the same leaderboard: Jais-ASR **22.85**,
Audar-ASR-V1-Turbo 23.17, Speechmatics Melia 1 23.51, ElevenLabs Scribe v2 24.54,
Cohere transcribe-arabic-07-2026 25.87, omniASR-LLM-7B 28.32, Qwen3-Omni-30B-A3B
30.71, Qwen3-ASR-1.7B 33.36, Voxtral-Small-24B 34.47.

**Read the table honestly.** Jais-ASR has the best macro-average, but it does not
win every set: Audar is ahead on SADA (28.92 vs 30.85), Cohere on Common Voice
(5.82 vs 8.56), and Speechmatics Melia on Casablanca (46.41 vs 46.56). The
conversational multi-dialect set (Casablanca) is roughly five times harder than
clean MSA read speech — that is the realistic ceiling for a noisy phone recording
with several dialects in one call.

These numbers are for the bf16 weights. The skill runs 4-bit NF4 to fit a 6 GB
GPU, so expect somewhat higher error rates than the table in practice.

## Languages and dialects

`ar` means "Arabic, any supported dialect"; `msa` pins Modern Standard Arabic.
Automatic detection is the model's default (the skill's `--language auto`).

| Setting | Language |
|---|---|
| `ar` | Arabic incl. all 19 dialects |
| `msa` | Modern Standard Arabic |
| `en` | English |
| `fr` | French |
| `es` | Spanish |
| `it` | Italian |

The 19 Arabic dialect codes:

| Code | Dialect | Code | Dialect | Code | Dialect |
|---|---|---|---|---|---|
| `DZ` | Algerian | `IQ` | Iraqi | `OM` | Omani |
| `BH` | Bahraini | `JO` | Jordanian | `PS` | Palestinian |
| `EG` | Egyptian | `KW` | Kuwaiti | `QA` | Qatari |
| `AE` | Emirati | `LB` | Lebanese | `SA` | Saudi |
| `LY` | Libyan | `MR` | Mauritanian | `SD` | Sudanese |
| `MA` | Moroccan | `SY` | Syrian | `TN` | Tunisian |
| `YE` | Yemeni | | | | |

The skill pins `--language EG` by default because its reference material is
Egyptian. Set `--language auto` to let the model identify the dialect, or any
code above to pin it.

## Code-switching

Arabic speech that drops into English, French, Spanish or Italian is transcribed
with the foreign spans kept in Latin script — enable it with `code_switch=True`
on the chat template, which the pipeline always does. It composes with
`diacritics=True` (not enabled by default here).

Also available but unused by this skill: per-transcript language/dialect
identification, and an explicit `[IFM-NoTranscript]` refusal (`refused=True`) when
there is no speech — the pipeline maps that to `(no speech detected)` instead of
letting the model invent text.

## How the skill calls it

- audio: 16 kHz mono, decoded by ffmpeg, enhanced, then cut into ≤ 30 s windows
  taken only from diarized speech regions
- generation: greedy (`do_sample=False`), `max_new_tokens=500`, `language=` as
  chosen by `--language`, `code_switch=True`
- output: `processor.post_process_transcription(...)`; refused windows become
  empty and render as `(no speech detected)`

## Citations

```bibtex
@misc{jaisasr2026,
  title  = {Introducing Jais-ASR: Arabic Speech Recognition for Every Dialect},
  author = {{IFM Team}},
  year   = {2026},
  url    = {https://huggingface.co/IFM/Jais-ASR},
}
```

- Model card: https://huggingface.co/IFM/Jais-ASR
- Leaderboard: https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard
