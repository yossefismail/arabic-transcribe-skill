# Setup — Arabic Transcribe

One-time environment preparation. Tested on: Windows 11, Python 3.12 (`py -3.12`), RTX 4050 6 GB, Ryzen 7. On macOS/Linux the same commands work with `python3.12` instead of `py -3.12` (untested, but the only Windows-specific code paths are guarded).

## Python environment

```sh
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
py -3.12 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cu128
py -3.12 -m pip install transformers==5.16.1 bitsandbytes soundfile imageio-ffmpeg huggingface-hub noisereduce
py -3.12 -m pip install --no-deps resemble-enhance   # neural denoiser (deepspeed dep is unbuildable on Windows; script stubs it)
py -3.12 -m pip install tabulate                     # imported by resemble_enhance at load time
```

No diarization package is needed: NVIDIA's `nemo-speech` runtime installs itself (see below).

Verify: `py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"` must print a `+cu128` build and `True`.

**Never let pip install a plain `torch` wheel (CPU) over the cu128 one.** If `torch.cuda.is_available()` is False, reinstall torch from the cu128 index.

## Audio enhancement (Resemble Enhance denoiser)

The primary denoise stage is [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) (MIT, pure PyTorch): mono 16 kHz in → denoised full-band 44.1 kHz out, runs on the GPU at roughly 10× realtime. Beats DeepFilterNet on ease-of-install here (DFN has no Windows py3.12 wheel and needs Rust) and beats spectral `noisereduce` on quality by a wide margin. The script handles the two Windows quirks itself:

1. **`PosixPath` tags in checkpoint YAMLs** — first run downloads the checkpoint, whose YAML was saved on Linux; loading it on Windows raises `NotImplementedError: cannot instantiate 'PosixPath'`. Fix (done in-script, Windows only): `pathlib.PosixPath = pathlib.WindowsPath` before importing.
2. **`deepspeed` import** — `resemble_enhance` imports deepspeed (training-only) at module level; it cannot be built on Windows/py3.12. On first import the script writes a minimal inference-only stub into a temp dir (`%TEMP%\arabic-transcribe-deps\deepspeed\`) and prepends it to `sys.path` — nothing is installed into your Python environment and no admin rights are needed. Content: `deepspeed/` with `DeepSpeedConfig`, `get_accelerator`, `DeepSpeedEngine`, `clip_grad_norm_` (the only symbols resemble_enhance touches). If that ever fails, that's the exact content to create manually and point `PYTHONPATH` at.

If `resemble_enhance` is missing or errors, the pipeline falls back to spectral `noisereduce` automatically — worse quality, same pipeline.

## Speaker diarization — NVIDIA Nemotron-3-Diarization (no account needed)

Default backend. [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
is NVIDIA's Sortformer V3 diarizer: ungated, OpenMDW-1.1 (commercial use allowed),
up to **8 speakers**, streaming or offline, and robust to overlap and far-field
audio.

It runs through **`nemo-speech`**, NVIDIA's native C++ runtime
([NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp), Apache-2.0). No
PyTorch, no NeMo Python stack, no Hugging Face login. The pipeline handles
everything on first run:

1. downloads the pinned **CPU build** (`v0.3.0`, ~8 MB) from the GitHub release
   matching your OS/arch, verifies the published SHA-256, and unpacks it to
   `~/.cache/arabic-transcribe/nemo-speech/0.3.0/`;
2. the CLI pulls the **102 MB** `Nemotron-3-Diarization.q8_0.gguf` into its own
   cache (`%LOCALAPPDATA%\NeMoSpeech\models` on Windows) and verifies its size
   and SHA-256;
3. diarization is written as RTTM and parsed back into speaker turns.

Overrides (all optional):

| Variable | Effect |
|---|---|
| `NEMO_SPEECH_BIN` | path to an existing `nemo-speech` binary (e.g. the Windows CUDA or Vulkan build) |
| `NEMO_DIAR_DEVICE` | `auto` (default), `cpu`, `cuda:0`, `vulkan:0` |
| `NEMO_DIAR_PRESET` | streaming preset; default `v3-offline` (large chunks, ~10× faster on short files, unlimited duration) |

The CPU build is deliberate: diarization is a small part of total runtime, and
keeping it off the GPU leaves the 6 GB card entirely to Jais-ASR. On a 57 s
4-speaker reference clip it takes ~2 s.

## Jais-ASR model (~11.4 GB, first run only)

`run_pipeline.py` downloads `IFM/Jais-ASR` (Apache-2.0) through huggingface_hub automatically. The repo ships custom modelling code that `transformers` executes (`trust_remote_code=True`), so the pipeline loads a **pinned commit** by default (`JAIS_REVISION`, currently `1323995fa189c479369a3929be0e95039eba5cba`). Set `JAIS_REVISION=main` if you deliberately want to track the hub.

**If the download stalls** (progress freezes at some MB/s burst then hangs with no error — the Xet backend on this network opens idle TLS connections that ignore timeouts), do NOT retry with huggingface_hub. Use the curl resume loop instead — it survives stalls by aborting slow connections and resuming by byte offset:

```sh
py -3.12 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
# then, for each of the 32 repo files (shard names:
# model-00001-of-00003.safetensors 4,986,988,032 B
# model-00002-of-00003.safetensors 4,981,009,992 B
# model-00003-of-00003.safetensors 1,468,086,312 B):
# loop: curl -C - -L -H "Authorization: Bearer hf_..." <file-url> -o <dest>
#       with --speed-limit 10240 --speed-time 20, up to 40 attempts per file
```

Then point the pipeline at the local dir: the scripts resolve `IFM/Jais-ASR`; pass env `JAIS_LOCAL_DIR=<path>` to force a local directory (checked first by run_pipeline.py via `local_files_only`).

Note: the HF cache keeps ~4.8 GB of redundant `.incomplete` blobs after a salvaged download — safe to delete `models--IFM--Jais-ASR` from the cache once a complete local copy exists.

## Runtime pitfalls

| Symptom | Cause / fix |
|---|---|
| `OSError: The paging file is too small (os error 1455)` during model load | Windows commit limit, not disk. Free RAM (close browser tabs), don't run pip concurrently, retry alone. |
| `ValueError: Some modules are dispatched on the CPU or the disk` | bitsandbytes rejecting the device map on ≤6 GB GPUs. Already solved: scripts pass `llm_int8_enable_fp32_cpu_offload=True`. |
| `nemo-speech` download/SHA failure | First use fetches the binary from the NVIDIA GitHub release. Check access to github.com, delete `~/.cache/arabic-transcribe/nemo-speech/`, or point `NEMO_SPEECH_BIN` at a local build. |
| Diarization merges two speakers or invents one | Sortformer infers the count (max 8). Tune the vendor preset via `NEMO_DIAR_PRESET`; very quiet or heavily overlapped speech is the usual cause. |
| Diarization output looks empty | nemo-speech accepts PCM WAV; the pipeline always writes 16 kHz mono. If the CLI rejects a preset it retries with the default pass automatically. |
| ffmpeg not found | Use `imageio_ffmpeg.get_ffmpeg_exe()` (bundled static binary). |
| `NotImplementedError: cannot instantiate 'PosixPath'` | Resemble Enhance checkpoint loading on Windows. Already solved: script aliases `pathlib.PosixPath = pathlib.WindowsPath` first. |
| `ModuleNotFoundError: No module named 'deepspeed'` | Already solved: script writes an inference-only stub to a temp dir and adds it to `sys.path` (see enhancement section). |
| `ImportError: Import tabulate failed` | `py -3.12 -m pip install tabulate`. |
| `RuntimeError: Tensor.item() cannot be called on meta tensors` during Jais load | RAM pressure made accelerate disk-dispatch quantized layers. Already solved: pipeline clears the denoiser cache + gc before loading, and passes an explicit `max_memory` budget so nothing lands on 'disk'. If it still appears, close RAM-heavy apps and retry. |

## Output notes

- The HTML is fully self-contained (audio embedded as base64 AAC 64k mono) — shareable as a single file, works offline.
- Within each 30 s Jais window, clause highlight timing is proportional to clause length — approximate but follows the audio closely.
- The TXT uses UTF-8 with BOM + RLM/LRM bidi marks so Arabic/English lines render correctly in Notepad.
