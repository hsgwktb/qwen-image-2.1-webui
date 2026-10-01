#!/usr/bin/env python3
"""
Qwen-Image-2.1 Uncensored — one-shot Colab bootstrap.

Provisions a ComfyUI backend plus the custom WebUI on the current Colab
runtime, then prints the URL to open.

Quantization is chosen from the GPU actually attached. On an L4 (24 GB) the
script selects Q8_0 for the diffusion transformer, which is the highest-quality
quantization that still leaves comfortable headroom for the int8 text encoder,
the VAE and sampling activations. Q4_K_M (the upstream author's recommendation)
is aimed at 8-12 GB consumer cards and would needlessly give up quality here.

Everything the WebUI serves comes from the GitHub repository; no web asset is
base64-embedded or inlined anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

HF_REPO = "0xSojalSec/Qwen-Image-2.1-Uncensored-HF"
GITHUB_REPO = os.environ.get("GITHUB_REPO", "hsgwktb/qwen-image-2.1-webui")
GITHUB_REF = os.environ.get("GITHUB_REF", "main")

WORK = Path("/content")
COMFY_DIR = WORK / "ComfyUI"
LIBS_DIR = Path("/content/libs")

COMFY_PORT = 8188
WEBUI_PORT = 7860

# diffusion transformer quantizations, largest first
DIT_QUANTS = {
    "BF16": "qwen-image-2.1-UC-BF16.gguf",
    "Q8_0": "qwen-image-2.1-UC-Q8_0.gguf",
    "Q6_K": "qwen-image-2.1-UC-Q6_K.gguf",
    "Q5_K_M": "qwen-image-2.1-UC-Q5_K_M.gguf",
    "Q4_K_M": "qwen-image-2.1-UC-Q4_K_M.gguf",
    "Q4_0": "qwen-image-2.1-UC-Q4_0.gguf",
}
DIT_BYTES = {
    "BF16": 14_230_272_800,
    "Q8_0": 7_591_557_920,
    "Q6_K": 5_876_556_576,
    "Q5_K_M": 5_221_284_640,
    "Q4_K_M": 4_604_558_112,
    "Q4_0": 4_151_573_280,
}

TEXT_ENCODERS = {
    "int8": "qwen3vl_8b_int8_convrot.safetensors",
    "bf16": "qwen3vl_8b_bf16.safetensors",
}
VAE_FILE = "qwen_image_2.1_vae_bf16.safetensors"

# dit quant -> text encoder, picked from available VRAM
VRAM_PLAN = [
    (40.0, "BF16", "bf16"),
    (20.0, "Q8_0", "int8"),
    (14.0, "Q6_K", "int8"),
    (10.0, "Q5_K_M", "int8"),
    (0.0, "Q4_K_M", "int8"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def sh(cmd: str, cwd: Path | None = None, check: bool = True) -> int:
    log(f"$ {cmd}")
    rc = subprocess.call(cmd, shell=True, cwd=str(cwd) if cwd else None)
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc


def pip(*args: str) -> None:
    sh(f"{sys.executable} -m pip install -q {(' '.join(args))}")


# --------------------------------------------------------------------------
# 1. hardware
# --------------------------------------------------------------------------

def detect_gpu() -> tuple[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            log("⚠ 没有检测到 CUDA GPU。请选择「运行时 → 更改运行时类型 → L4 GPU」。")
            return "cpu", 0.0
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log(f"✓ GPU: {name}  ({total:.1f} GB VRAM)")
        return name, total
    except Exception as exc:  # noqa: BLE001
        log(f"⚠ 无法检测 GPU: {exc}")
        return "unknown", 0.0


def choose_plan(gpu_name: str, vram_gb: float) -> tuple[str, str]:
    requested = os.environ.get("QWEN_QUANT", "").strip().upper()
    if requested:
        if requested not in DIT_QUANTS:
            raise SystemExit(f"QWEN_QUANT 无效: {requested}，可选 {list(DIT_QUANTS)}")
        te = "bf16" if requested == "BF16" else "int8"
        log(f"→ 使用手动指定的量化: {requested}")
        return requested, te

    for threshold, quant, te in VRAM_PLAN:
        if vram_gb >= threshold:
            log(
                f"→ 显存 {vram_gb:.1f} GB，选择 DiT={quant} "
                f"({DIT_BYTES[quant] / 1024 ** 3:.2f} GB) + 文本编码器={te}"
            )
            return quant, te

    return "Q4_K_M", "int8"


# --------------------------------------------------------------------------
# 2. ComfyUI + ComfyUI-GGUF
# --------------------------------------------------------------------------

def install_comfyui() -> None:
    if not COMFY_DIR.exists():
        log("\n[1/5] 克隆 ComfyUI…")
        sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}")
    else:
        log("\n[1/5] ComfyUI 已存在，跳过克隆")

    log("[1/5] 安装 ComfyUI 依赖（保留 Colab 自带的 torch）…")
    req = COMFY_DIR / "requirements.txt"
    if req.exists():
        keep = [
            ln
            for ln in req.read_text().splitlines()
            if ln.strip()
            and not ln.lower().startswith(("torch", "torchvision", "torchaudio"))
        ]
        tmp = Path("/tmp/comfy-req.txt")
        tmp.write_text("\n".join(keep))
        sh(f"{sys.executable} -m pip install -q -r {tmp}")

    nodes = COMFY_DIR / "custom_nodes"
    nodes.mkdir(parents=True, exist_ok=True)

    # The leejet fork carries native Qwen-Image 2.1 support; the older
    # city96/ComfyUI-GGUF raises "Unknown model architecture!" for this model.
    gguf_dir = nodes / "ComfyUI-GGUF"
    if not gguf_dir.exists():
        log("[1/5] 安装 ComfyUI-GGUF (leejet fork)…")
        sh(f"git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF.git {gguf_dir}")
    greq = gguf_dir / "requirements.txt"
    if greq.exists():
        sh(f"{sys.executable} -m pip install -q -r {greq}")
    else:
        pip("gguf")


def install_webui_deps() -> None:
    log("[2/5] 安装 WebUI 服务端依赖…")
    pip("fastapi", "uvicorn", "httpx", "websocket-client")


# --------------------------------------------------------------------------
# 3. models
# --------------------------------------------------------------------------

def hf_download(remote: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(remote).name
    if target.exists() and target.stat().st_size > 0:
        log(f"   ↳ 已存在: {target.name} ({target.stat().st_size / 1024 ** 3:.2f} GB)")
        return target

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=remote,
        local_dir=str(dest_dir),
        resume_download=True,
    )
    return Path(path)


def download_models(quant: str, text_encoder: str) -> None:
    log(f"\n[3/5] 下载模型 (DiT={quant}, TE={text_encoder})…")
    dit_file = DIT_QUANTS[quant]
    te_file = TEXT_ENCODERS[text_encoder]

    diffs = COMFY_DIR / "models" / "diffusion_models"
    hf_download(dit_file, diffs)

    # ComfyUI builds differ on whether the GGUF loader scans "unet" or
    # "diffusion_models"; keep the weights in both.
    legacy = COMFY_DIR / "models" / "unet"
    legacy.mkdir(parents=True, exist_ok=True)
    link = legacy / dit_file
    if not link.exists():
        try:
            link.symlink_to(diffs / dit_file)
        except OSError:
            shutil.copy2(diffs / dit_file, link)

    hf_download(f"text_encoders/{te_file}", COMFY_DIR / "models" / "text_encoders")
    hf_download(f"vae/{VAE_FILE}", COMFY_DIR / "models" / "vae")

    log("✓ 模型就绪")


# --------------------------------------------------------------------------
# 4. services
# --------------------------------------------------------------------------

def wait_for(url: str, timeout: float = 300.0, name: str = "service") -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(2.0)
    log(f"✗ {name} 在 {timeout:.0f}s 内没有就绪")
    return False


def start_comfyui(lowvram: bool = False) -> subprocess.Popen:
    log("\n[4/5] 启动 ComfyUI…")
    logfile = open("/content/comfyui.log", "w")
    extra = " --lowvram" if lowvram else ""
    cmd = (
        f"{sys.executable} main.py --listen 127.0.0.1 --port {COMFY_PORT} "
        f"--disable-auto-launch --dont-print-server{extra}"
    )
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(COMFY_DIR), stdout=logfile, stderr=subprocess.STDOUT
    )
    if not wait_for(f"http://127.0.0.1:{COMFY_PORT}/system_stats", 420, "ComfyUI"):
        log("—— ComfyUI 日志尾部 ——")
        log(subprocess.run(
            "tail -n 40 /content/comfyui.log", shell=True, capture_output=True, text=True
        ).stdout)
        raise RuntimeError("ComfyUI 启动失败")
    log("✓ ComfyUI 已就绪")
    return proc


def start_webui(quant: str, text_encoder: str) -> subprocess.Popen:
    log("[5/5] 启动 WebUI…")
    env = os.environ.copy()
    env.update(
        {
            "COMFY_URL": f"http://127.0.0.1:{COMFY_PORT}",
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_REF": GITHUB_REF,
            "ASSET_SOURCE": os.environ.get("ASSET_SOURCE", "proxy"),
            "PORT": str(WEBUI_PORT),
            "UNET_NAME": DIT_QUANTS[quant],
            "CLIP_NAME": TEXT_ENCODERS[text_encoder],
            "VAE_NAME": VAE_FILE,
            "QUANT_LABEL": quant,
        }
    )
    logfile = open("/content/webui.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "/content/server.py"],
        stdout=logfile,
        stderr=subprocess.STDOUT,
        env=env,
        cwd="/content",
    )
    if not wait_for(f"http://127.0.0.1:{WEBUI_PORT}/api/config", 180, "WebUI"):
        log(subprocess.run(
            "tail -n 40 /content/webui.log", shell=True, capture_output=True, text=True
        ).stdout)
        raise RuntimeError("WebUI 启动失败")
    log("✓ WebUI 已就绪")
    return proc


def fetch_webui_code() -> None:
    """Pull server.py from GitHub so the notebook itself stays free of logic."""
    import urllib.error
    import urllib.request

    sources = [
        f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_REF}/server.py",
        f"https://cdn.jsdelivr.net/gh/{GITHUB_REPO}@{GITHUB_REF}/server.py",
    ]
    last: Exception | None = None
    for url in sources:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                Path("/content/server.py").write_bytes(r.read())
            log(f"✓ 已从 GitHub 获取 server.py ({url.split('/')[2]})")
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"   ↳ {url.split('/')[2]} 失败: {exc}")
    raise RuntimeError(f"无法从 GitHub 获取 server.py: {last}")


def public_url() -> str | None:
    try:
        from google.colab.output import eval_js  # type: ignore

        return eval_js(f"google.colab.kernel.proxyPort({WEBUI_PORT})")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------

def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    log("=" * 66)
    log("Qwen-Image-2.1 Uncensored · Colab 部署")
    log("=" * 66)

    gpu_name, vram_gb = detect_gpu()
    quant, text_encoder = choose_plan(gpu_name, vram_gb)

    install_comfyui()
    install_webui_deps()
    download_models(quant, text_encoder)
    fetch_webui_code()
    start_comfyui(lowvram=(vram_gb and vram_gb < 12))
    start_webui(quant, text_encoder)

    url = public_url()

    log("")
    log("=" * 66)
    log("✅ 部署完成")
    log(f"   模型      : {HF_REPO}")
    log(f"   DiT 量化  : {quant}  ({DIT_BYTES[quant] / 1024 ** 3:.2f} GB)")
    log(f"   文本编码器: {TEXT_ENCODERS[text_encoder]}")
    log(f"   显存      : {vram_gb:.1f} GB")
    log(f"   ComfyUI   : http://127.0.0.1:{COMFY_PORT}")
    log(f"   WebUI     : http://127.0.0.1:{WEBUI_PORT}")
    if url:
        log(f"   🔗 打开   : {url}")
    else:
        log("   🔗 在笔记本中运行下方单元获取代理 URL")
    log("=" * 66)


if __name__ == "__main__":
    main()
