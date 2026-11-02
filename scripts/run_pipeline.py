"""Arabic Transcribe pipeline: NVIDIA Sortformer diarization -> Jais-ASR (4-bit)
on speech turns only -> synced HTML player with embedded enhanced audio.

Usage:
    py -3.12 run_pipeline.py <input audio> [--outdir DIR] [--quant 4bit|8bit]
                             [--max-seconds N] [--no-enhance]

Flow:
    1. decode to 16 kHz mono
    2. enhance audio: neural denoise (Resemble Enhance, 16k -> 44.1 kHz),
       then speech EQ, compressor/AGC, loudness normalize; ASR/diarization
       run on the enhanced audio resampled back to 16 kHz
    3. diarize the FULL file with NVIDIA Nemotron-3-Diarization (Sortformer V3,
       up to 8 speakers) through the nemo-speech CLI — its speaker turns double
       as the speech map, so long noise tails are skipped without a separate VAD.
       No Hugging Face account or token is involved.
    4. merge turns into speech blocks; transcribe each block with Jais-ASR
    5. write synced HTML player (audio embedded, live highlight, click-to-seek,
       speaker colors) + plain-text transcript with bidi marks
    6. (agent-side, after this script) add_summary.py appends a summary with
       an action-points checklist to the end of both outputs
"""
import argparse, base64, hashlib, json, os, platform, re, shutil, subprocess, sys, tarfile, tempfile, time, urllib.request, zipfile
import numpy as np

def die(msg):
    print("ERROR:", msg, flush=True)
    sys.exit(1)

def fmt(sec):
    sec = max(0.0, sec)
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------------------------------------------------------- audio decode
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

def decode(path, tmpdir):
    wav = os.path.join(tmpdir, "audio_16k.wav")
    cmd = [ffmpeg_exe(), "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(wav):
        die("ffmpeg decode failed:\n" + (r.stderr or "")[-1500:])
    import soundfile as sf
    data, sr = sf.read(wav, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

# ---------------------------------------------------------------- enhancement
def _ensure_deepspeed_stub():
    """resemble_enhance imports deepspeed (training-only) at module level, and
    deepspeed cannot be built on Windows/py3.12. Write a minimal inference-only
    stub to a temp dir and put it on sys.path — the user's Python environment is
    never modified and no admin rights are needed."""
    try:
        import deepspeed  # noqa: F401
        return
    except ImportError:
        pass
    root = os.path.join(tempfile.gettempdir(), "arabic-transcribe-deps")
    base = os.path.join(root, "deepspeed")
    files = {
        "__init__.py": '''"""Minimal deepspeed stub (inference-only) for resemble_enhance."""
import torch


class DeepSpeedConfig:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("deepspeed stub: training is not available")


class DeepSpeedEngine:
    pass


class _Accelerator:
    def device_name(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_device(self):
        return torch.device(self.device_name())


def get_accelerator():
    return _Accelerator()
''',
        "accelerator.py": '''import torch


class _Accelerator:
    def device_name(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_device(self):
        return torch.device(self.device_name())


def get_accelerator():
    return _Accelerator()
''',
        os.path.join("runtime", "__init__.py"): "",
        os.path.join("runtime", "engine.py"): "class DeepSpeedEngine:\n    pass\n",
        os.path.join("runtime", "utils.py"): '''import torch


def clip_grad_norm_(*args, **kwargs):
    return torch.nn.utils.clip_grad_norm_(*args, **kwargs)
''',
    }
    for rel, src in files.items():
        p = os.path.join(base, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
    if root not in sys.path:
        sys.path.insert(0, root)
    print("  created deepspeed stub in temp dir (inference-only)", flush=True)

def resemble_denoise(x, sr=16000, device=None):
    """Neural denoising via Resemble Enhance (MIT, pure PyTorch).
    1-D mono tensor at 16 kHz in -> denoised full-band audio at 44.1 kHz out.
    Windows quirks handled here: checkpoint YAMLs carry PosixPath tags, and
    the deepspeed training-only import needs a stub."""
    import pathlib
    if sys.platform == "win32":
        # checkpoint yamls were pickled on Linux; on Windows the PosixPath tag in
        # them needs a class the OS can instantiate
        pathlib.PosixPath = pathlib.WindowsPath
    _ensure_deepspeed_stub()
    import torch
    from resemble_enhance.enhancer.inference import denoise
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out, out_sr = denoise(torch.from_numpy(x.astype(np.float32)), sr, device=device)
    return out.cpu().numpy().astype(np.float32), int(out_sr)

def resample_16k(x, sr):
    """Polyphase resample to 16 kHz for ASR/diarization."""
    import scipy.signal as sps
    if sr == 16000:
        return x.astype(np.float32)
    g = np.gcd(int(sr), 16000)
    return sps.resample_poly(x, 16000 // g, sr // g).astype(np.float32)
def _peaking_biquad(sr, f0, gain_db, q=1.0):
    """RBJ peaking EQ biquad as SOS."""
    import scipy.signal as sps
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b0, b1, b2 = 1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A
    a0, a1, a2 = 1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

def enhance(x, sr):
    """Make speech louder and clearer for ASR + listening:
    neural denoise (Resemble Enhance; spectral NR fallback) -> high-pass rumble
    removal -> mud cut at 250 Hz -> presence boost at 2.5 kHz -> chunked
    compressor/AGC -> loudness normalize.
    Returns (audio, sr) — sr becomes 44.1 kHz when the neural denoiser runs
    (it outputs full-band 44.1 kHz)."""
    import scipy.signal as sps

    # 1) denoise: neural model (primary), spectral subtraction (fallback)
    try:
        x, sr = resemble_denoise(x, sr)
        print(f"  denoise: Resemble Enhance (neural) -> {sr} Hz", flush=True)
    except Exception as e:
        print(f"  denoise: neural unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
        try:
            import noisereduce as nr
            x = nr.reduce_noise(y=x, sr=sr, stationary=True, prop_decrease=0.75,
                                n_fft=1024, hop_length=256)
            print("  noise reduction: spectral fallback applied", flush=True)
        except ImportError:
            print("  noise reduction: skipped (pip install noisereduce)", flush=True)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # free denoiser workspace before diarization/Jais
    except Exception:
        pass

    # 3) remove rumble / handling noise below the speech band
    sos = sps.butter(4, 90, btype="highpass", fs=sr, output="sos")
    x = sps.sosfiltfilt(sos, x)

    # 4) speech-band EQ: cut low-mid mud, lift presence/intelligibility
    x = sps.sosfilt(_peaking_biquad(sr, 250, -3.0, q=0.8), x)
    x = sps.sosfilt(_peaking_biquad(sr, 2500, 5.0, q=1.0), x)
    x = sps.sosfilt(_peaking_biquad(sr, 5000, 2.0, q=1.2), x)

    # 5) chunked compressor/AGC: lift quiet speech, tame peaks.
    #    Frame-by-frame loop: never materializes an (n, frame) matrix, so
    #    hour-long files at 44.1 kHz stay well under 1 GB extra RAM
    #    (vectorized fancy-indexing needed ~3.5 GB and OOMs on long files).
    frame, hop = int(0.25 * sr), int(0.125 * sr)
    n = (len(x) - frame) // hop + 1
    if n >= 4:
        x = x.astype(np.float32, copy=False)
        rms_db = np.empty(n, dtype=np.float64)
        for i in range(n):
            seg = x[i * hop:i * hop + frame]
            rms_db[i] = 20 * np.log10(
                np.sqrt(np.mean(seg ** 2, dtype=np.float64)) + 1e-10)
        # 3:1 above -20 dBFS, up to +10 dB lift for quiet parts
        over = np.clip(rms_db, -60, 0)
        gain_db = np.minimum(0.0, (-20 + (over - (-20)) / 3) - over)
        gain_db = np.clip(gain_db, -12.0, 10.0)
        gain = 10 ** (gain_db / 20)
        g = np.interp(np.arange(len(x), dtype=np.float32),
                      np.arange(n) * hop + frame // 2, gain).astype(np.float32)
        x = np.clip(x * g, -1.0, 1.0)

    # 6) loudness normalize: scale speech RMS to -14 dBFS, then soft-limit peaks
    active = np.abs(x) > 0.01
    if np.any(active):
        rms = float(np.sqrt(np.mean(x[active] ** 2)) + 1e-10)
        x = x * (10 ** (-14.0 / 20)) / rms
    return np.clip(x, -0.89, 0.89).astype(np.float32), sr  # ≈ -1 dBFS ceiling

# ---------------------------------------------------------------- diarization
# NVIDIA Nemotron-3-Diarization (Sortformer V3): ungated, OpenMDW-1.1, up to 8
# speakers, streaming or offline. Runs locally through NVIDIA's nemo-speech CLI
# (Apache-2.0, native C++ — no PyTorch, no Hugging Face login, no token).
NEMO_SPEECH_VERSION = "0.3.0"
NEMO_SPEECH_RELEASE = ("https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/"
                       f"v{NEMO_SPEECH_VERSION}")
NEMO_DIAR_MODEL = "nvidia/Nemotron-3-Diarization"


def _nemo_speech_asset():
    """Release asset for this OS/arch. The CPU build is ~8 MB and needs no CUDA;
    set NEMO_SPEECH_BIN to any other build (e.g. the Windows CUDA one)."""
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86-64": "x86_64", "x86_64": "x86_64",
            "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)
    if sys.platform == "win32":
        os_name, ext = "windows", "zip"
    elif sys.platform == "darwin":
        os_name, ext = "macos", "tar.gz"
    elif sys.platform.startswith("linux"):
        os_name, ext = "linux", "tar.gz"
    else:
        die(f"no nemo-speech build for platform {sys.platform}")
    return f"nemo-speech-{NEMO_SPEECH_VERSION}-{os_name}-{arch}-cpu.{ext}"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest, what):
    print(f"  downloading {what} ...", flush=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)


def ensure_nemo_speech():
    """Return the nemo-speech executable, installing the pinned CPU build once.
    Resolution order: NEMO_SPEECH_BIN, PATH, then ~/.cache/arabic-transcribe.
    Release assets are verified against the published SHA-256 before they run."""
    env = os.environ.get("NEMO_SPEECH_BIN")
    if env:
        if os.path.isfile(env):
            return env
        die(f"NEMO_SPEECH_BIN={env} does not exist")
    found = shutil.which("nemo-speech") or shutil.which("nemo-speech.exe")
    if found:
        return found

    root = os.path.join(os.path.expanduser("~"), ".cache", "arabic-transcribe",
                        "nemo-speech", NEMO_SPEECH_VERSION)
    exe_name = "nemo-speech.exe" if sys.platform == "win32" else "nemo-speech"
    for base, _dirs, files in os.walk(root):
        if exe_name in files:
            return os.path.join(base, exe_name)

    os.makedirs(root, exist_ok=True)
    asset = _nemo_speech_asset()
    url = f"{NEMO_SPEECH_RELEASE}/{asset}"
    archive = os.path.join(root, asset)
    if not os.path.isfile(archive):
        _download(url, archive, f"nemo-speech {NEMO_SPEECH_VERSION} ({asset})")
        _download(url + ".sha256", archive + ".sha256", "nemo-speech checksum")
        with open(archive + ".sha256", encoding="utf-8") as f:
            expected = f.read().split()[0].strip().lower()
        actual = _sha256_file(archive)
        if actual != expected:
            os.remove(archive)
            die("nemo-speech failed its SHA-256 check and was deleted "
                f"(expected {expected[:12]}…, got {actual[:12]}…)")

    extract = os.path.join(root, "dist")
    print("  unpacking nemo-speech ...", flush=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(extract)
    else:
        with tarfile.open(archive, "r:gz") as t:
            try:
                t.extractall(extract, filter="data")   # py3.12+
            except TypeError:
                t.extractall(extract)
    for base, _dirs, files in os.walk(extract):
        if exe_name in files:
            path = os.path.join(base, exe_name)
            if sys.platform != "win32":
                os.chmod(path, 0o755)
            return path
    die(f"nemo-speech archive {asset} did not contain {exe_name}")


def _parse_rttm(path):
    """RTTM: SPEAKER <file> <chan> <start> <dur> <ortho> <type> <name> <conf>."""
    turns = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) < 8 or p[0] != "SPEAKER":
                continue
            try:
                start, dur = float(p[3]), float(p[4])
            except ValueError:
                continue
            turns.append({"start": round(start, 3), "end": round(start + dur, 3),
                          "speaker": p[7]})
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


def diarize_nemo(x, sr, tmpdir):
    """Diarize with NVIDIA Nemotron-3-Diarization via the nemo-speech CLI."""
    import soundfile as sf
    exe = ensure_nemo_speech()
    wav = os.path.join(tmpdir, "diarization.wav")
    rttm = os.path.join(tmpdir, "diarization.rttm")
    sf.write(wav, x, sr)

    # v3-offline is a large-chunk streaming pass: ~10x faster than the default
    # on short files and still unlimited in duration. Override with NEMO_DIAR_PRESET.
    preset = os.environ.get("NEMO_DIAR_PRESET", "v3-offline")
    device = os.environ.get("NEMO_DIAR_DEVICE", "auto")
    print(f"Diarizing with {NEMO_DIAR_MODEL} (nemo-speech {NEMO_SPEECH_VERSION}) ...",
          flush=True)
    cmd = [exe, "diarize", wav, "--format", "rttm", "--recording-id", "audio",
           "--model", NEMO_DIAR_MODEL, "--device", device, "-o", rttm]
    r = subprocess.run(cmd + ["--preset", preset] if preset else cmd,
                       capture_output=True, text=True)
    if r.returncode != 0 and preset:
        print(f"  preset '{preset}' rejected, retrying with the default streaming pass",
              flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(rttm):
        die("nemo-speech diarization failed:\n" + (r.stderr or r.stdout or "")[-1500:])

    turns = _parse_rttm(rttm)
    if not turns:
        die("diarization produced no speaker turns (no speech, or the model "
            "rejected the audio)")
    return turns

def speech_blocks(turns, sr, min_speech=3.0, merge_gap=15.0):
    """Merge speaker turns into speech blocks; drop blocks with too little talk.
    Returns [(start, end)] in seconds."""
    runs = [(t["start"], t["end"]) for t in turns if t["end"] - t["start"] > 0.2]
    if not runs:
        return []
    runs.sort()
    blocks = [[runs[0][0], runs[0][1]]]
    for s, e in runs[1:]:
        if s - blocks[-1][1] <= merge_gap:
            blocks[-1][1] = max(blocks[-1][1], e)
        else:
            blocks.append([s, e])
    out = []
    for s, e in blocks:
        talk = sum(min(e, te) - max(s, ts) for ts, te in runs if min(e, te) > max(s, ts))
        if talk >= min_speech:
            out.append((max(0.0, s - 1.0), e + 1.0))
    return out

def windows_from_blocks(blocks, win=30.0):
    segs = []
    for a, b in blocks:
        s = a
        while s < b - 0.5:
            e = min(s + win, b)
            segs.append((s, e))
            s = e
    return segs

# ---------------------------------------------------------------- Jais (4-bit)
JAIS_MODEL = "IFM/Jais-ASR"
# IFM/Jais-ASR ships custom modelling code that transformers executes
# (trust_remote_code=True), so the skill pins an audited commit instead of
# tracking whatever `main` contains today. Override with JAIS_REVISION=main.
JAIS_REVISION = os.environ.get(
    "JAIS_REVISION", "1323995fa189c479369a3929be0e95039eba5cba")

JAIS_KNOWN_DIRS = [
    os.path.expanduser("~/.local/share/opencode/models/jais-asr"),
    os.path.expanduser("~/.cache/jais-asr"),
]

def _jais_dir_complete(d):
    """True if d holds a complete Jais-ASR snapshot: config + weight index +
    every shard the index references, each > 1 GB. Guards against the classic
    'cache looks present but 2 of 3 shards are missing' trap."""
    idx = os.path.join(d, "model.safetensors.index.json")
    if not (os.path.isfile(os.path.join(d, "config.json")) and os.path.isfile(idx)):
        return False
    try:
        with open(idx, encoding="utf-8") as f:
            shards = set(json.load(f)["weight_map"].values())
    except Exception:
        return False
    return all(os.path.isfile(os.path.join(d, s))
               and os.path.getsize(os.path.join(d, s)) > 1e9 for s in shards)

def resolve_jais_model():
    """Pick the Jais-ASR source: JAIS_LOCAL_DIR env, then known local
    locations, then the HF hub. Returns (source, is_local_dir). Local copies
    win because hub downloads of the ~11.4 GB model stall on some networks."""
    candidates, env = [], os.environ.get("JAIS_LOCAL_DIR")
    if env:
        candidates.append(env)
    candidates += JAIS_KNOWN_DIRS
    for d in candidates:
        if not os.path.isdir(d):
            continue
        if _jais_dir_complete(d):
            return d, True
        if d == env:
            die(f"JAIS_LOCAL_DIR={d} is set but its copy is INCOMPLETE "
                "(missing/undersized shards). Finish it via the curl salvage in "
                "references/setup.md, point the var at a complete dir, or unset it.")
    return JAIS_MODEL, False

def _mem_budget():
    """Device-map budget sized to the actual GPU.

    The 4-bit model must not offload layers to CPU: on this stack bitsandbytes
    cannot rebuild an offloaded module's quant_state (meta-tensor crash). So the
    GPU budget is simply "everything fits" for the card in use.

    6 GB cards keep the verified 5.3 GiB that holds all 449 Linear4bit layers on
    the GPU. Larger cards get free VRAM minus a working margin, so they stop
    shovelling layers to CPU for no reason. JAIS_GPU_MEM / JAIS_CPU_MEM override.
    """
    import torch
    if not torch.cuda.is_available():
        return {"cpu": os.environ.get("JAIS_CPU_MEM", "12GiB")}
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024 ** 3
    try:
        free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
    except Exception:
        free_gib = total_gib
    if total_gib <= 7.5:                 # the platform this was tuned on
        gib = 5.3
    else:
        gib = free_gib - 0.8
    gib = max(3.0, min(gib, total_gib - 0.3))
    return {0: os.environ.get("JAIS_GPU_MEM") or f"{gib:.1f}GiB",
            "cpu": os.environ.get("JAIS_CPU_MEM", "9GiB")}


def jais_load(quant="4bit", source=None, local_dir=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    src = source or JAIS_MODEL
    t0 = time.time()
    print(f"Loading Jais-ASR ({quant}) from {src} ...", flush=True)
    lo = {"local_files_only": True} if local_dir else {"revision": JAIS_REVISION}
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True, **lo)
    processor.tokenizer.padding_side = "left"
    kw = dict(llm_int8_enable_fp32_cpu_offload=True)  # REQUIRED on <=6 GB GPUs
    bnb = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, **kw)
           if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True, **kw))
    # explicit budget: if accelerate runs out of CPU RAM it dispatches layers to
    # 'disk' (meta device), which bitsandbytes quant_state cannot do -> crash
    max_mem = _mem_budget()
    print(f"  device map budget: {max_mem}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        src, trust_remote_code=True, device_map="auto", quantization_config=bnb,
        max_memory=max_mem, **lo).eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    return model, processor

def jais_transcribe(model, processor, x, sr, segs):
    import torch
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio"}]}],
        tokenize=False, add_generation_prompt=True, code_switch=True)  # English stays Latin script
    lines = []
    for s, e in segs:
        inputs = processor(text=[prompt], audio=[x[int(s * sr):int(e * sr)]],
                           sampling_rate=[sr], padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=500, language="EG")
        res = processor.post_process_transcription(out, input_lengths=inputs["input_ids"].shape[1])[0]
        text = "" if res.refused else re.sub(r"\s+", " ", (res.text or "")).strip()
        lines.append((s, e, text))
        print(f"  [{fmt(s)}-{fmt(e)}] {len(text)} chars", flush=True)
    return lines

# ---------------------------------------------------------------- outputs
RLM, LRM = "\u200f", "\u200e"

def build_outputs(base, in_path, lines, turns, m4a_b64, a, b):
    labels = {}
    for t in sorted(turns, key=lambda x: x["start"]):
        if t["speaker"] not in labels:
            labels[t["speaker"]] = f"Speaker {chr(ord('A') + len(labels))}"

    def speaker_at(sec):
        for t in turns:
            if t["start"] <= sec < t["end"]:
                return labels[t["speaker"]]
        return "Speaker ?"

    def split_clauses(txt):
        return [p for p in (q.strip() for q in re.split(r"(?<=[.؟!،؛?,:])\s+", txt)) if p]

    spk_list = sorted(set(labels.values()))

    # --- HTML player ---
    body = []
    for s, e, txt in lines:
        body.append(f'    <div class="win"><div class="meta" dir="ltr">[{fmt(s)}&thinsp;&ndash;&thinsp;{fmt(e)}]</div>')
        if not txt:
            body.append('      <p class="txt" dir="auto"><i>(no speech detected)</i></p></div>')
            continue
        clauses = split_clauses(txt)
        total = sum(len(c) for c in clauses) or 1
        dur, acc = e - s, 0.0
        body.append('      <p class="txt" dir="auto">')
        for c in clauses:
            cs = s + dur * acc / total
            acc += len(c)
            ce = s + dur * acc / total
            spk = speaker_at((cs + ce) / 2).split()[-1]
            body.append(f'<span class="chunk s{spk}" data-start="{cs:.2f}" data-end="{ce:.2f}">{esc(c)} </span>')
        body.append("</p></div>")

    legend = "".join(f'<span class="dot" style="background:{col}"></span>{spk}'
                     for spk, col in zip(spk_list, ["#0b6623", "#1c5d99", "#a34a00", "#7a3e9d", "#8a6d00"]))

    html = """<!DOCTYPE html>
<html lang="ar">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — transcript</title>
<style>
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', 'Noto Naskh Arabic', 'Geeza Pro', system-ui, sans-serif;
       background: #faf9f6; color: #1c1b18; margin: 0; padding: 0 0 3rem; }
.player { position: sticky; top: 0; z-index: 5; background: #fffdf8ee; backdrop-filter: blur(6px);
          border-bottom: 1px solid #e7e4db; padding: .7rem 1rem; }
.player audio { width: 100%; max-width: 680px; display: block; margin: 0 auto; }
.hdr { max-width: 760px; margin: 1.2rem auto .4rem; padding: 0 1rem; }
.hdr h1 { font-size: 1.15rem; margin: 0 0 .2rem; }
.hdr .sub { color: #6b675e; font-size: .82rem; }
.legend { max-width: 760px; margin: .4rem auto 1rem; padding: 0 1rem; font-size: .82rem; color: #6b675e; }
.legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 6px 0 14px; }
main { max-width: 760px; margin: 0 auto; padding: 0 1rem; }
.win { background: #fff; border: 1px solid #e7e4db; border-radius: 12px; padding: .8rem 1.1rem; margin: .7rem 0; }
.meta { font-size: .75rem; color: #8a857a; margin-bottom: .3rem; letter-spacing: .02em; }
.txt { font-size: 1.12rem; line-height: 2.05; margin: 0; }
.chunk { cursor: pointer; border-radius: 5px; padding: 1px 2px; transition: background .15s; }
.chunk:hover { background: #eee9dc; }
.sA { border-bottom: 2px solid #0b662355; } .sB { border-bottom: 2px solid #1c5d9955; }
.sC { border-bottom: 2px solid #a34a0055; } .sD { border-bottom: 2px solid #7a3e9d55; }
.sE { border-bottom: 2px solid #8a6d0055; }
.chunk.on.sA { background: #0b66231c; } .chunk.on.sB { background: #1c5d991c; }
.chunk.on.sC { background: #a34a001c; } .chunk.on.sD { background: #7a3e9d1c; }
.chunk.on.sE { background: #8a6d001c; }
.hint { text-align: center; color: #8a857a; font-size: .8rem; margin: .8rem 0 0; }
</style>
</head>
<body>
<div class="player"><audio id="au" controls preload="auto" src="data:audio/mp4;base64,__AUDIO__"></audio>
<div class="hint" dir="ltr">click any text to jump there &middot; the current clause highlights as it plays</div></div>
<div class="hdr">
<h1>__TITLE__</h1>
<div class="sub" dir="ltr">Jais-ASR (4-bit) &middot; NVIDIA Nemotron-3-Diarization &middot; speech __A__&ndash;__B__ &middot; Arabic kept as spoken, English in Latin script</div>
</div>
<div class="legend" dir="ltr">Speaker colors: __LEGEND__</div>
<main>
__BODY__
</main>
<script>
const au = document.getElementById('au');
const chunks = Array.from(document.querySelectorAll('.chunk'));
let last = null;
chunks.forEach(c => c.addEventListener('click', () => {
  au.currentTime = parseFloat(c.dataset.start) + 0.01;
  if (au.paused) au.play();
}));
au.addEventListener('timeupdate', () => {
  const t = au.currentTime;
  let act = null;
  for (const c of chunks) {
    const on = t >= parseFloat(c.dataset.start) && t < parseFloat(c.dataset.end);
    c.classList.toggle('on', on);
    if (on) act = c;
  }
  if (act && act !== last) { act.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); last = act; }
});
</script>
</body>
</html>"""
    html = (html.replace("__AUDIO__", m4a_b64)
                .replace("__TITLE__", esc(os.path.basename(in_path)))
                .replace("__A__", fmt(a)).replace("__B__", fmt(b))
                .replace("__LEGEND__", legend)
                .replace("__BODY__", "\n".join(body)))
    html_path = base + " - arabic transcript.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # --- TXT with bidi marks ---
    out = [f"TRANSCRIPT — {os.path.basename(in_path)}",
           "(Jais-ASR 4-bit + NVIDIA Nemotron-3-Diarization; English in Latin script)",
           f"Speech region: {fmt(a)}-{fmt(b)} | speakers: {', '.join(spk_list)}", "=" * 70, ""]
    for s, e, txt in lines:
        out.append(LRM + f"[{fmt(s)}-{fmt(e)}]")
        if txt:
            out.append(RLM + txt)
        else:
            out.append(LRM + "(no speech detected)")
        out.append("")
    txt_path = base + " - arabic transcript.txt"
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(out))
    return html_path, txt_path, spk_list

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir")
    ap.add_argument("--quant", choices=["4bit", "8bit"], default="4bit")
    ap.add_argument("--no-enhance", action="store_true", help="skip audio enhancement (high-pass, NR, EQ, compression)")
    ap.add_argument("--max-seconds", type=float, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        die(f"input not found: {args.input}")

    # resolve the model source up front — fail fast on an incomplete copy
    # instead of after diarization (or worse, mid-hub-download)
    jais_src, jais_local = resolve_jais_model()
    if jais_local:
        print(f"Jais-ASR: verified local copy at {jais_src}", flush=True)
    else:
        print("Jais-ASR: no verified local copy — will download ~11.4 GB from "
              "the HF hub (on networks where this stalls, see "
              "references/setup.md for the curl salvage)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        print("Decoding audio ...", flush=True)
        x, sr = decode(args.input, tmp)
        if args.max_seconds:
            x = x[: int(args.max_seconds * sr)]
        dur = len(x) / sr
        print(f"total: {dur:.0f}s", flush=True)

        player_src = None   # (audio, sr) at full quality for the player
        if not args.no_enhance:
            print("Enhancing audio (neural denoise, speech EQ, compression) ...", flush=True)
            y, sr_y = enhance(x, sr)
            player_src = (y, sr_y)
            x = resample_16k(y, sr_y)   # ASR/diarization input
            sr = 16000
            dur = len(x) / sr
            print(f"enhanced: {dur:.0f}s, ASR input resampled to 16 kHz", flush=True)

        turns = diarize_nemo(x, sr, tmp)
        print(f"Diarization: {len(turns)} speaker turns", flush=True)

        blocks = speech_blocks(turns, sr)
        if not blocks:
            die("No speech detected in this file.")
        a, b = blocks[0][0], blocks[-1][1]
        segs = windows_from_blocks(blocks)
        total_speech = sum(e - s for s, e in blocks)
        print(f"Speech blocks: {[(fmt(s), fmt(e)) for s, e in blocks]}", flush=True)
        print(f"Transcribing {len(segs)} windows ({total_speech:.0f}s of speech) ...", flush=True)

        # free every cached byte before the 4-bit load: RAM pressure makes
        # accelerate disk-dispatch layers, which bitsandbytes cannot handle
        import gc
        try:
            import resemble_enhance.enhancer.inference as _ri
            _ri.load_enhancer.cache_clear()   # drop the in-process denoiser model
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        model, processor = jais_load(args.quant, jais_src, jais_local)
        lines = jais_transcribe(model, processor, x, sr, segs)
        del model   # free VRAM/RAM before writing outputs

        # player audio: enhanced full-rate file trimmed to first speech -1s .. last +1s
        import soundfile as sf
        py_sig, py_sr = player_src if player_src is not None else (x, sr)
        wav_enh = os.path.join(tmp, "enhanced.wav")
        sf.write(wav_enh, py_sig, py_sr)
        m4a = os.path.join(tmp, "player.m4a")
        cmd = [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, a - 1.0):.2f}",
               "-to", f"{min(dur, b + 1.0):.2f}", "-i", wav_enh,
               "-ac", "1", "-ar", "44100", "-c:a", "aac", "-b:a", "64k", m4a]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(m4a):
            die("ffmpeg encode failed:\n" + (r.stderr or "")[-1500:])
        m4a_b64 = base64.b64encode(open(m4a, "rb").read()).decode()
        # also keep the enhanced audio as an independent file next to the outputs
        import shutil
        enh_base = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.input))[0]) if args.outdir else os.path.splitext(args.input)[0]
        enh_path = enh_base + " - enhanced audio.m4a"
        try:
            shutil.copyfile(m4a, enh_path)
            print("SAVED", enh_path, flush=True)
        except Exception as e:
            print("WARN could not copy enhanced audio:", e, flush=True)
        # player timeline starts at trim point: shift all timestamps
        trim = max(0.0, a - 1.0)
        lines = [(s - trim, e - trim, txt) for s, e, txt in lines]
        turns = [{"start": t["start"] - trim, "end": t["end"] - trim, "speaker": t["speaker"]}
                 for t in turns if t["end"] > trim]
        a, b = a - trim, b - trim

    base = os.path.splitext(args.input)[0] if not args.outdir else os.path.join(args.outdir, os.path.splitext(os.path.basename(args.input))[0])
    html_path, txt_path, spk_list = build_outputs(base, args.input, lines, turns, m4a_b64, a, b)
    print("SAVED", html_path, flush=True)
    print("SAVED", txt_path, flush=True)
    print(f"DONE elapsed={time.time()-t0:.0f}s speakers={spk_list}", flush=True)

if __name__ == "__main__":
    main()
