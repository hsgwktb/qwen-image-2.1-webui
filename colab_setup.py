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
import re
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
        sh(f"git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git {COMFY_DIR}")
    else:
        log("\n[1/5] ComfyUI 已存在，跳过克隆")

    log("[1/5] 安装 ComfyUI 依赖（保留 Colab 自带的 torch）…")
    req = COMFY_DIR / "requirements.txt"
    if req.exists():
        # Colab already ships the three CUDA torch packages, so skip exactly
        # those. Match the distribution name precisely: `torchsde` (a real
        # ComfyUI dependency) merely shares the "torch" prefix and must be
        # installed, otherwise ComfyUI dies on import.
        skip = {"torch", "torchvision", "torchaudio"}
        keep = []
        for line in req.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            pkg = re.split(r"[<>=!~\[\s;]", s, 1)[0].strip().lower()
            if pkg not in skip:
                keep.append(s)
        tmp = Path("/tmp/comfy-req.txt")
        tmp.write_text("\n".join(keep))
        log(f"   ↳ {len(keep)} 个依赖（跳过 Colab 自带的 {', '.join(sorted(skip))}）")
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
    """
    Fetch one file from the model repo and place it directly in `dest_dir`.

    hf_hub_download preserves the repo-relative path under local_dir, so a
    remote name like "text_encoders/foo.safetensors" would land in
    <dest_dir>/text_encoders/foo.safetensors. ComfyUI only scans the top level
    of each model folder, so the file has to be flattened afterwards — that
    mismatch is what makes ComfyUI report a mysterious
    prompt_outputs_failed_validation.
    """
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(remote).name
    nested = dest_dir / remote

    if target.exists() and target.stat().st_size > 0:
        log(f"   ↳ 已存在: {target.name} ({target.stat().st_size / 1024 ** 3:.2f} GB)")
        return target

    # a previous run left it in the nested repo layout — just relocate it
    if nested != target and nested.exists() and nested.stat().st_size > 0:
        shutil.move(str(nested), str(target))
        log(f"   ↳ 已就位: {target.name} ({target.stat().st_size / 1024 ** 3:.2f} GB)")
        return target

    path = Path(
        hf_hub_download(repo_id=HF_REPO, filename=remote, local_dir=str(dest_dir))
    )
    if path != target and path.exists():
        shutil.move(str(path), str(target))
    if not target.exists():
        raise RuntimeError(f"下载后找不到文件: {target}")

    log(f"   ↳ 已下载: {target.name} ({target.stat().st_size / 1024 ** 3:.2f} GB)")
    return target


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

def wait_for(url: str, timeout: float = 300.0, name: str = "service", proc=None) -> bool:
    """Poll a URL until it answers, bailing out early if the process died."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            log(f"✗ {name} 进程已退出（returncode={proc.returncode}），不再等待")
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(2.0)
    log(f"✗ {name} 在 {timeout:.0f}s 内没有就绪")
    return False


def dump_log(path: str, lines: int = 60) -> None:
    log(f"—— {path} 末尾 {lines} 行 ——")
    try:
        out = subprocess.run(
            f"tail -n {lines} {path}", shell=True, capture_output=True, text=True
        ).stdout
        log(out or "(空)")
    except Exception as exc:  # noqa: BLE001
        log(f"(无法读取日志: {exc})")


def start_comfyui(lowvram: bool = False) -> subprocess.Popen:
    log("\n[4/5] 启动 ComfyUI…")
    logfile = open("/content/comfyui.log", "w")
    extra = " --lowvram" if lowvram else ""
    cmd = (
        f"{sys.executable} main.py --listen 127.0.0.1 --port {COMFY_PORT} "
        f"--disable-auto-launch{extra}"
    )
    log(f"   $ {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(COMFY_DIR), stdout=logfile, stderr=subprocess.STDOUT
    )
    if not wait_for(
        f"http://127.0.0.1:{COMFY_PORT}/system_stats", 420, "ComfyUI", proc=proc
    ):
        dump_log("/content/comfyui.log", 60)
        raise RuntimeError("ComfyUI 启动失败，详见上方 /content/comfyui.log 末尾")
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
            "COMFY_INPUT_DIR": str(COMFY_DIR / "input"),
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
    if not wait_for(
        f"http://127.0.0.1:{WEBUI_PORT}/api/config", 180, "WebUI", proc=proc
    ):
        dump_log("/content/webui.log", 40)
        raise RuntimeError("WebUI 启动失败，详见上方 /content/webui.log 末尾")
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
# public access
#
# Colab's own proxyPort URL (*.prod.colab.dev) is not reachable from every
# network. A Cloudflare quick tunnel gives a plain https://…trycloudflare.com
# address that works without a proxy, so use that for external access.
# --------------------------------------------------------------------------

def start_tunnel(port: int) -> str | None:
    import re as _re
    import urllib.request

    binary = "/content/cloudflared"
    if not os.path.exists(binary):
        url = (
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64"
        )
        log(f"   下载 cloudflared …")
        try:
            urllib.request.urlretrieve(url, binary)
            os.chmod(binary, 0o755)
        except Exception as exc:  # noqa: BLE001
            log(f"   ✗ cloudflared 下载失败: {exc}")
            return None

    logf = open("/content/cloudflared.log", "w")
    subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=logf,
        stderr=subprocess.STDOUT,
    )

    for _ in range(90):
        time.sleep(1)
        try:
            text = Path("/content/cloudflared.log").read_text(errors="replace")
        except Exception:  # noqa: BLE001
            continue
        m = _re.search(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com", text)
        if m:
            return m.group(0)
    dump_log("/content/cloudflared.log", 30)
    return None


# --------------------------------------------------------------------------
# diagnostics — ComfyUI rejects a workflow with a terse validation error, so
# report exactly which nodes/files the running server actually exposes.
# --------------------------------------------------------------------------

def diagnose(info: dict) -> None:
    import urllib.request

    for cls in (
        "UnetLoaderGGUF",
        "UnetLoaderGGUFAdvanced",
        "UNETLoader",
        "TextEncodeQwenImage21",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "EmptySD3LatentImage",
        "PrimitiveStringMultiline",
        "KSampler",
        "VAELoader",
        "CLIPLoader",
    ):
        log(f"   节点 {cls:26s}: {'有' if cls in info else '缺失'}")

    def opts(cls: str, field: str):
        spec = (info.get(cls, {}).get("input", {}).get("required", {}) or {}).get(field)
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            return spec[0]
        return None

    te_types = opts("CLIPLoader", "type")
    if te_types:
        log(f"   CLIPLoader.type 支持: {te_types}")

    if "TextEncodeQwenImage21" in info:
        req = (info["TextEncodeQwenImage21"].get("input", {}).get("required", {}) or {})
        log(f"   TextEncodeQwenImage21 必填输入: {list(req.keys())}")

    # dump the exact input names/types for the nodes involved in image editing
    for cls in ("TextEncodeQwenImage21", "BatchImagesNode", "ImageBatch", "LoadImage"):
        if cls not in info:
            continue
        spec = info[cls].get("input", {})
        for section in ("required", "optional"):
            for name, definition in (spec.get(section) or {}).items():
                kind = definition[0] if isinstance(definition, list) and definition else None
                if isinstance(kind, list):
                    kind = f"combo[{len(kind)}]"
                log(f"   {cls}.{name} ({section}): {kind}")

    # what does ComfyUI actually see on disk for the GGUF loader?
    for cls in ("UnetLoaderGGUF", "UNETLoader"):
        if cls not in info:
            continue
        for field in ("unet_name", "ckpt_name"):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{COMFY_PORT}/object_info/{cls}", timeout=30
                ) as r:
                    d = json.loads(r.read())
                names = (
                    d.get(cls, {})
                    .get("input", {})
                    .get("required", {})
                    .get(field, [[None]])[0]
                )
                if isinstance(names, list):
                    log(f"   {cls}.{field} 可选 ({len(names)}): {names[:8]}")
                break
            except Exception:  # noqa: BLE001
                continue


# --------------------------------------------------------------------------
# self-test — proves the whole ComfyUI graph actually renders an image
# --------------------------------------------------------------------------

def _submit_and_wait(port: int, payload: dict, timeout: float, label: str) -> bool:
    import json as _json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/generate",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            res = _json.loads(r.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        log(f"   ✗ {label}提交失败 HTTP {exc.code}")
        log("   —— ComfyUI 完整报错 ——")
        # wrap so the notebook output does not clip long lines
        for i in range(0, min(len(body), 3000), 110):
            log("     " + body[i : i + 110])
        dump_log("/content/comfyui.log", 40)
        return False
    except Exception as exc:  # noqa: BLE001
        log(f"   ✗ {label}提交失败: {exc}")
        return False

    pid = res.get("prompt_id")
    log(f"   已提交{label}任务 {pid}（首次会加载权重到显存，可能 1-3 分钟）")

    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        time.sleep(5)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/result/{pid}", timeout=30
            ) as r:
                st = _json.loads(r.read())
        except Exception:  # noqa: BLE001
            continue
        status = st.get("status")
        if status == "done":
            imgs = st.get("images") or []
            log(f"   ✓ {label}通过：生成 {len(imgs)} 张图像")
            for im in imgs[:1]:
                log(f"     {im.get('filename')}")
            return True
        if status == "error":
            log(f"   ✗ {label}失败: {str(st.get('error'))[:900]}")
            dump_log("/content/comfyui.log", 40)
            return False
        cur = f"{status} {st.get('step', '')}/{st.get('total', '')}".strip()
        if cur != last:
            log(f"     …{cur}")
            last = cur

    log(f"   ✗ {label}超时")
    dump_log("/content/comfyui.log", 40)
    return False


def self_test(port: int, timeout: float = 1200.0) -> bool:
    """Text-to-image: proves the sampling graph renders."""
    return _submit_and_wait(
        port,
        {
            "prompt": "a single red cube on a white table, studio lighting, photorealistic",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "seed": 12345,
            "batch": 1,
        },
        timeout,
        "文生图自检",
    )


# the two reference images used by the official Comfy-Org image-edit template
EDIT_SAMPLES = [
    (
        "portrait_model_denim.png",
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/"
        "input/portrait_model_denim.png",
    ),
    (
        "clothing_light_blue_denim_shirt.png",
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/"
        "input/clothing_light_blue_denim_shirt.png",
    ),
]


def edit_self_test(port: int, timeout: float = 1800.0) -> bool:
    """Image editing: image_1 is the target, image_2 a reference."""
    import json as _json
    import urllib.request

    names: list[str] = []
    for fname, url in EDIT_SAMPLES:
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            up = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/upload?filename={fname}",
                data=data,
                headers={"Content-Type": "image/png"},
            )
            with urllib.request.urlopen(up, timeout=120) as r:
                meta = _json.loads(r.read())
            names.append(meta["name"])
            size = f"{meta.get('width')}×{meta.get('height')}"
            note = "（已等比缩放）" if meta.get("resized") else "（原始分辨率）"
            log(f"   ↳ {fname}: {size}{note}")
        except Exception as exc:  # noqa: BLE001
            log(f"   ✗ 获取示例图失败 {fname}: {exc}")
            return False

    log(f"   官方示例图已就位: {names}")

    return _submit_and_wait(
        port,
        {
            "prompt": (
                "Replace the costume of the character in <image1> "
                "with the clothing shown in <image2>"
            ),
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "seed": 4242,
            "batch": 1,
            "images": names,
        },
        timeout,
        "图像编辑自检",
    )


# --------------------------------------------------------------------------

def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    log("=" * 66)
    log("Qwen-Image-2.1 Uncensored · Colab 部署")
    log("=" * 66)

    # Re-running the cell is the normal way to iterate, so clear whatever a
    # previous attempt left bound to 8188 / 7860 before starting again.
    subprocess.run(
        "pkill -f 'ComfyUI/main.py'; pkill -f '/content/server.py'; pkill -f cloudflared",
        shell=True,
    )
    time.sleep(3)

    gpu_name, vram_gb = detect_gpu()
    quant, text_encoder = choose_plan(gpu_name, vram_gb)

    install_comfyui()
    install_webui_deps()
    download_models(quant, text_encoder)
    fetch_webui_code()
    start_comfyui(lowvram=bool(vram_gb and vram_gb < 12))
    start_webui(quant, text_encoder)

    log("\n[诊断] 检查 ComfyUI 实际暴露的节点与模型文件…")
    try:
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{COMFY_PORT}/object_info", timeout=120
        ) as r:
            diagnose(json.loads(r.read()))
    except Exception as exc:  # noqa: BLE001
        log(f"   诊断失败: {exc}")

    log("\n[自检] 文生图：生成一张测试图，验证采样链路…")
    ok = self_test(WEBUI_PORT)

    log("\n[自检] 图像编辑：用官方示例图做一次换装编辑…")
    edit_ok = edit_self_test(WEBUI_PORT)

    log("\n[隧道] 建立公网访问地址…")
    tunnel = start_tunnel(WEBUI_PORT)
    proxy = public_url()

    log("")
    log("=" * 66)
    if ok and edit_ok:
        log("✅ 部署完成（文生图 + 图像编辑自检均通过）")
    elif ok:
        log("⚠️ 部署完成：文生图正常，图像编辑自检未通过（见上方错误）")
    else:
        log("⚠️ 部署完成，但自检未通过（见上方错误）")
    log(f"   模型      : {HF_REPO}")
    log(f"   DiT 量化  : {quant}  ({DIT_BYTES[quant] / 1024 ** 3:.2f} GB)")
    log(f"   文本编码器: {TEXT_ENCODERS[text_encoder]}")
    log(f"   显存      : {vram_gb:.1f} GB")
    log(f"   图像编辑  : {'可用（最多 10 张参考图）' if edit_ok else '不可用'}")
    log(f"   ComfyUI   : http://127.0.0.1:{COMFY_PORT}")
    log(f"   WebUI     : http://127.0.0.1:{WEBUI_PORT}")
    if tunnel:
        log(f"   🔗 公网地址: {tunnel}")
    if proxy:
        log(f"   🔗 Colab代理: {proxy}")
    if not tunnel and not proxy:
        log("   🔗 在笔记本中运行下方单元获取代理 URL")
    log("=" * 66)


if __name__ == "__main__":
    main()
