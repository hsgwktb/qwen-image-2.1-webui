#!/usr/bin/env python3
"""
Qwen-Image-2.1 Uncensored — WebUI server.

Serves a custom single-page front-end and drives a local ComfyUI instance
through its HTTP API.

The front-end files are NOT embedded in this file and are NOT base64 data
URIs: they are fetched from the project's GitHub repository at startup and
re-served from here, so the notebook stays tiny and the web assets have a
single source of truth on GitHub.

Environment:
  COMFY_URL        ComfyUI base URL            (default http://127.0.0.1:8188)
  GITHUB_REPO      owner/name holding web/*   (default hsgwktb/qwen-image-2.1-webui)
  GITHUB_REF       branch or tag              (default main)
  ASSET_SOURCE     proxy | github             (default proxy)
  PORT             port for this server       (default 7860)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "hsgwktb/qwen-image-2.1-webui")
GITHUB_REF = os.environ.get("GITHUB_REF", "main")
ASSET_SOURCE = os.environ.get("ASSET_SOURCE", "proxy").lower()
PORT = int(os.environ.get("PORT", "7860"))

# ComfyUI's LoadImage reads from its own input directory; uploads land there.
INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/content/ComfyUI/input"))
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_REFERENCE_IMAGES = 10

RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_REF}/web"
CDN_BASE = f"https://cdn.jsdelivr.net/gh/{GITHUB_REPO}@{GITHUB_REF}/web"

ASSET_FILES = ("index.html", "style.css", "app.js")
ASSETS: dict[str, bytes] = {}
ASSET_ORIGIN = RAW_BASE

DEFAULT_NEGATIVE = (
    "低分辨率, 低质量, 最差质量, 模糊, 变形, 畸形, 多余的手指, "
    "文字, 水印, 签名"
)

app = FastAPI(title="Qwen-Image-2.1 Uncensored WebUI")
http = httpx.Client(timeout=httpx.Timeout(30.0, read=180.0))

# prompt_id -> {"progress": float, "step": int, "total": int, "node": str}
PROGRESS: dict[str, dict[str, Any]] = {}
PROGRESS_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# asset acquisition — the web files live on GitHub, never in the notebook
# --------------------------------------------------------------------------

def fetch_assets() -> None:
    """Pull the front-end bundle from GitHub. Falls back to the jsDelivr CDN."""
    global ASSET_ORIGIN

    for base in (RAW_BASE, CDN_BASE):
        try:
            got: dict[str, bytes] = {}
            for name in ASSET_FILES:
                r = httpx.get(f"{base}/{name}", timeout=30.0, follow_redirects=True)
                r.raise_for_status()
                got[name] = r.content
            ASSETS.update(got)
            ASSET_ORIGIN = base
            print(f"[assets] loaded {len(got)} files from {base}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[assets] {base} failed: {exc}", file=sys.stderr, flush=True)

    print("[assets] FATAL: could not fetch front-end from GitHub", file=sys.stderr, flush=True)


def index_html() -> str:
    """Render index.html, optionally pointing the asset tags at GitHub itself."""
    html = ASSETS.get("index.html", b"").decode("utf-8", "replace")
    if ASSET_SOURCE == "github":
        html = html.replace('href="/assets/style.css"', f'href="{ASSET_ORIGIN}/style.css"')
        html = html.replace('src="/assets/app.js"', f'src="{ASSET_ORIGIN}/app.js"')
    return html


# --------------------------------------------------------------------------
# ComfyUI introspection
# --------------------------------------------------------------------------

def comfy_get(path: str, **params: Any) -> Any:
    r = http.get(f"{COMFY_URL}{path}", params=params or None)
    r.raise_for_status()
    return r.json()


def comfy_post(path: str, payload: dict) -> Any:
    r = http.post(f"{COMFY_URL}{path}", json=payload)
    r.raise_for_status()
    return r.json()


def object_info() -> dict:
    return comfy_get("/object_info")


def gpu_snapshot() -> dict:
    info: dict[str, Any] = {}
    try:
        stats = comfy_get("/system_stats")
        for dev in stats.get("devices", []):
            if dev.get("type") == "cuda" or dev.get("name"):
                info["gpu_name"] = dev.get("name")
                total = dev.get("vram_total") or 0
                free = dev.get("vram_free") or 0
                if total:
                    info["vram_total_gb"] = total / (1024 ** 3)
                if free:
                    info["vram_free_gb"] = free / (1024 ** 3)
                break
    except Exception as exc:  # noqa: BLE001
        info["comfy_error"] = f"ComfyUI 未就绪: {type(exc).__name__}"
    return info


_EDIT_CAP: dict[str, Any] = {"value": None, "batch": None, "ts": 0.0}


def editing_capabilities() -> tuple[bool, str | None]:
    """(encode node present?, node used to merge several references)."""
    now = time.time()
    if _EDIT_CAP["value"] is None or now - _EDIT_CAP["ts"] > 300:
        try:
            info = object_info()
            _EDIT_CAP["value"] = "TextEncodeQwenImage21" in info
            _EDIT_CAP["batch"] = image_batch_class(info)
            _EDIT_CAP["ts"] = now
        except Exception:  # noqa: BLE001
            return False, None
    return bool(_EDIT_CAP["value"]), _EDIT_CAP["batch"]


def editing_available() -> bool:
    return editing_capabilities()[0]


# --------------------------------------------------------------------------
# workflow construction — schema-driven so it survives ComfyUI version drift
# --------------------------------------------------------------------------

# ComfyUI encodes an input as ["INT", {"default": 1024, ...}] for primitives and
# as [[choice, choice], {...}] for combos. These strings are type markers, never
# values — feeding one to the server yields
# "invalid literal for int() with base 10: 'INT'".
_TYPE_MARKERS = {
    "INT", "FLOAT", "STRING", "BOOLEAN", "IMAGE", "LATENT", "MASK", "MODEL",
    "CLIP", "VAE", "CONDITIONING", "CONTROL_NET", "SAMPLER", "SIGMAS",
    "GUIDER", "NOISE", "AUDIO", "VIDEO", "WEBCAM", "IMAGEUPLOAD", "ANY", "*",
}


def _default_for(spec: Any) -> Any:
    """Pull a usable default out of an /object_info input definition."""
    if not isinstance(spec, list) or not spec:
        return None
    first = spec[0]

    if isinstance(first, list):                     # combo of allowed values
        return first[0] if first else None

    # A dict in position 1 means this is a typed input such as
    # ["INT", {"default": 1024}] — the type name is not a value.
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else None
    if opts is not None:
        return opts.get("default")                  # may legitimately be absent

    if isinstance(first, str) and first.upper() in _TYPE_MARKERS:
        return None                                 # ["IMAGEUPLOAD"] with no opts

    if isinstance(first, (int, float, bool, str)):
        return first

    return None


def make_node(class_type: str, info: dict, links: dict, overrides: dict) -> dict:
    """
    Build a ComfyUI node, filling every required input from the live schema and
    only overriding the ones we actually care about. Anything we do not
    explicitly set keeps the server's own default.
    """
    if class_type not in info:
        raise RuntimeError(f"ComfyUI 缺少节点: {class_type}")

    schema = info[class_type].get("input", {})
    inputs: dict[str, Any] = {}
    for section in ("required", "optional"):
        for name, spec in (schema.get(section) or {}).items():
            if name in links:
                inputs[name] = links[name]
            elif name in overrides:
                inputs[name] = overrides[name]
            else:
                dflt = _default_for(spec)
                if dflt is not None:
                    inputs[name] = dflt
    return {"class_type": class_type, "inputs": inputs}


def image_batch_class(info: dict) -> str | None:
    """
    Node used to merge several reference images into the single batched IMAGE
    that TextEncodeQwenImage21 expects.

    ImageBatch (plain image1/image2, and it rescales image2 to match image1) is
    preferred because its contract is unambiguous. BatchImagesNode is the node
    the official template uses, but it takes a variable list of dotted inputs
    whose exact names differ between builds.
    """
    for c in ("ImageBatch", "ImageBatchMulti", "BatchImagesNode"):
        if c in info:
            return c
    return None


def pick(info: dict, *candidates: str) -> str:
    for c in candidates:
        if c in info:
            return c
    raise RuntimeError("ComfyUI 缺少以下节点之一: " + ", ".join(candidates))


def combo_values(info: dict, cls: str, field: str) -> list | None:
    """The allowed values of a combo input, straight from the live schema."""
    spec = (info.get(cls, {}).get("input", {}).get("required", {}) or {}).get(field)
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return spec[0]
    return None


def pick_combo(info: dict, cls: str, field: str, preferred: str, contains: str | None = None) -> str:
    """Return `preferred` when the server accepts it, else the closest match."""
    vals = combo_values(info, cls, field)
    if not vals:
        return preferred
    if preferred in vals:
        return preferred
    if contains:
        for v in vals:
            if contains in str(v):
                return v
    return preferred


def require_value(info: dict, cls: str, field: str, value: str, label: str) -> None:
    """
    Fail loudly when a model file is not where ComfyUI looks for it. Without
    this the server only reports a generic prompt_outputs_failed_validation.
    """
    vals = combo_values(info, cls, field)
    if vals is not None and value not in vals:
        raise RuntimeError(
            f"{label} 不在 ComfyUI 的可选列表中: {value!r}。"
            f"当前 {cls}.{field} 可选: {vals[:8]}"
        )


def make_string_node(info: dict, wf: dict, node_id: str, text: str):
    """
    Create a string-primitive node, discovering its input name from the live
    schema (ComfyUI has shipped this as both `value` and `text`). Returns a link
    reference, or None when the node type is unavailable.
    """
    cls = "PrimitiveStringMultiline"
    if cls not in info:
        return None
    schema = info[cls].get("input", {})
    names = list((schema.get("required") or {}).keys()) + list((schema.get("optional") or {}).keys())
    if not names:
        return None
    name = "value" if "value" in names else names[0]
    wf[node_id] = {"class_type": cls, "inputs": {name: text}}
    return [node_id, 0]


def build_workflow(req: dict, info: dict, prefer_modern: bool = True) -> dict:
    """Assemble a text-to-image graph for Qwen-Image-2.1 (GGUF)."""
    width = int(req.get("width", 1024))
    height = int(req.get("height", 1024))
    steps = int(req.get("steps", 25))
    cfg = float(req.get("cfg", 1.0))
    seed = int(req.get("seed", 0))
    batch = int(req.get("batch", 1))
    sampler = req.get("sampler", "euler")
    scheduler = req.get("scheduler", "simple")
    prompt = req.get("prompt", "")
    negative = req.get("negative") or DEFAULT_NEGATIVE

    unet_name = req.get("unet") or os.environ.get("UNET_NAME", "qwen-image-2.1-UC-Q8_0.gguf")
    clip_name = req.get("clip") or os.environ.get("CLIP_NAME", "qwen3vl_8b_int8_convrot.safetensors")
    vae_name = req.get("vae") or os.environ.get("VAE_NAME", "qwen_image_2.1_vae_bf16.safetensors")

    wf: dict[str, Any] = {}

    # --- model ------------------------------------------------------------
    unet_cls = next(
        (c for c in ("UnetLoaderGGUF", "UnetLoaderGGUFAdvanced") if c in info), None
    )
    if unet_cls is None:
        if unet_name.lower().endswith(".gguf"):
            raise RuntimeError(
                "ComfyUI 没有注册 GGUF 加载节点（UnetLoaderGGUF）。"
                "请检查 custom_nodes/ComfyUI-GGUF 是否安装成功、gguf 包是否可用。"
            )
        unet_cls = pick(info, "UNETLoader")

    unet_req = info[unet_cls].get("input", {}).get("required", {}) or {}
    unet_field = "unet_name" if "unet_name" in unet_req else "ckpt_name"
    require_value(info, unet_cls, unet_field, unet_name, "DiT 模型")
    wf["1"] = make_node(unet_cls, info, {}, {unet_field: unet_name})

    # --- text encoder -----------------------------------------------------
    te_type = pick_combo(info, "CLIPLoader", "type", "qwen_image", contains="qwen")
    require_value(info, "CLIPLoader", "clip_name", clip_name, "文本编码器")
    wf["2"] = make_node(
        "CLIPLoader", info, {}, {"clip_name": clip_name, "type": te_type}
    )

    # --- vae --------------------------------------------------------------
    require_value(info, "VAELoader", "vae_name", vae_name, "VAE")
    wf["3"] = make_node("VAELoader", info, {}, {"vae_name": vae_name})

    # --- reference images (image editing) ---------------------------------
    images = [
        str(x) for x in (req.get("images") or []) if str(x).strip()
    ][:MAX_REFERENCE_IMAGES]

    # --- latent -----------------------------------------------------------
    latent_cls = pick(info, "EmptyLatentImage", "EmptySD3LatentImage")
    wf["4"] = make_node(
        latent_cls,
        info,
        {},
        {"width": width, "height": height, "batch_size": batch},
    )
    latent_ref: list[Any] = ["4", 0]

    # --- conditioning + sampling -----------------------------------------
    # Qwen-Image-2.1 ships a dedicated encoder node that returns both
    # conditionings at once; older ComfyUI builds only have CLIPTextEncode.
    if prefer_modern and "TextEncodeQwenImage21" in info:
        links: dict[str, Any] = {"clip": ["2", 0], "vae": ["3", 0]}
        overrides: dict[str, Any] = {}
        pos_ref = make_string_node(info, wf, "6", prompt)
        neg_ref = make_string_node(info, wf, "7", negative)
        if pos_ref and neg_ref:
            links["prompt"] = pos_ref
            links["negative_prompt"] = neg_ref
        else:
            # no string-primitive node: feed the literals straight in
            overrides["prompt"] = prompt
            overrides["negative_prompt"] = negative

        # image_1 is the edit target, image_2..10 are references.
        load_refs: list[Any] = []
        for idx, name in enumerate(images, start=1):
            nid = str(30 + idx)
            wf[nid] = make_node("LoadImage", info, {}, {"image": name})
            load_refs.append([nid, 0])

        # The node takes ONE batched IMAGE, so the references get merged first.
        # Node ids 50+ are reserved for the merge chain.
        if load_refs:
            if len(load_refs) == 1:
                merged: Any = load_refs[0]
            else:
                cls = image_batch_class(info)
                if cls is None:
                    raise RuntimeError(
                        "需要 BatchImagesNode 或 ImageBatch 节点来合并多张参考图，"
                        "当前 ComfyUI 两者都没有。"
                    )
                if cls == "BatchImagesNode":
                    wf["50"] = {
                        "class_type": cls,
                        "inputs": {
                            f"images.image{i}": r for i, r in enumerate(load_refs)
                        },
                    }
                    merged = ["50", 0]
                else:
                    cur = load_refs[0]
                    for i, r in enumerate(load_refs[1:], start=51):
                        wf[str(i)] = {
                            "class_type": cls,
                            "inputs": {"image1": cur, "image2": r},
                        }
                        cur = [str(i), 0]
                    merged = cur

            # make_node keeps only the keys the node's schema actually declares,
            # so offering both spellings covers either ComfyUI variant.
            links["images"] = merged
            for i, r in enumerate(load_refs, start=1):
                links[f"images.image_{i}"] = r

        wf["5"] = make_node("TextEncodeQwenImage21", info, links, overrides)
        pos, neg = ["5", 0], ["5", 1]

        if images:
            # In edit mode the canvas is derived from image_1 rather than from a
            # blank latent — see the official image-edit template note
            # "canvas comes from the encode latent (image_1)".
            latent_ref = ["5", 2]
    elif images:
        raise RuntimeError(
            "图像编辑需要 ComfyUI 提供 TextEncodeQwenImage21 节点，当前实例没有该节点。"
        )
    else:
        wf["5"] = make_node("CLIPTextEncode", info, {"clip": ["2", 0]}, {"text": prompt})
        wf["6"] = make_node("CLIPTextEncode", info, {"clip": ["2", 0]}, {"text": negative})
        pos, neg = ["5", 0], ["6", 0]

    wf["10"] = make_node(
        "KSampler",
        info,
        {
            "model": ["1", 0],
            "positive": pos,
            "negative": neg,
            "latent_image": latent_ref,
        },
        {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    )

    wf["11"] = make_node("VAEDecode", info, {"samples": ["10", 0], "vae": ["3", 0]}, {})
    wf["12"] = make_node("SaveImage", info, {"images": ["11", 0]}, {"filename_prefix": "qwen21"})

    return wf


# --------------------------------------------------------------------------
# live step progress via ComfyUI's websocket
# --------------------------------------------------------------------------

def ws_progress_listener() -> None:
    try:
        import websocket  # type: ignore
    except Exception:  # noqa: BLE001
        print("[progress] websocket-client unavailable; progress bar disabled", flush=True)
        return

    ws_url = COMFY_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
    while True:
        try:
            ws = websocket.create_connection(ws_url, timeout=30)
            print("[progress] connected to ComfyUI websocket", flush=True)
            while True:
                raw = ws.recv()
                if not raw:
                    continue
                msg = json.loads(raw)
                mtype, data = msg.get("type"), msg.get("data", {})
                pid = data.get("prompt_id")
                if not pid:
                    continue
                with PROGRESS_LOCK:
                    slot = PROGRESS.setdefault(pid, {})
                    if mtype == "progress":
                        cur, tot = data.get("value", 0), data.get("max", 1) or 1
                        slot.update(
                            progress=5 + 90 * cur / tot,
                            step=cur,
                            total=tot,
                            node=data.get("node") or slot.get("node", ""),
                        )
                    elif mtype in ("executing", "execution_start"):
                        slot.setdefault("progress", 5.0)
                        slot["node"] = data.get("node") or ""
                    elif mtype in ("execution_success", "execution_error", "execution_cached"):
                        slot["progress"] = 100.0
        except Exception:  # noqa: BLE001
            time.sleep(2.0)


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    if not ASSETS:
        fetch_assets()
    return HTMLResponse(index_html())


@app.get("/assets/{name}")
def asset(name: str) -> Response:
    if name not in ASSETS:
        return Response("not found", status_code=404)
    ctype = {
        "style.css": "text/css; charset=utf-8",
        "app.js": "application/javascript; charset=utf-8",
        "index.html": "text/html; charset=utf-8",
    }.get(name, "application/octet-stream")
    return Response(ASSETS[name], media_type=ctype, headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def api_config() -> JSONResponse:
    payload: dict[str, Any] = {
        "asset_source": ASSET_ORIGIN,
        "asset_mode": ASSET_SOURCE,
        "github_repo": GITHUB_REPO,
        "github_ref": GITHUB_REF,
        "comfy_url": COMFY_URL,
        "input_dir": str(INPUT_DIR),
        "editing_available": editing_capabilities()[0],
        "image_batch_node": editing_capabilities()[1],
        "max_reference_images": MAX_REFERENCE_IMAGES,
        "unet": os.environ.get("UNET_NAME", "qwen-image-2.1-UC-Q8_0.gguf"),
        "text_encoder": os.environ.get("CLIP_NAME", "qwen3vl_8b_int8_convrot.safetensors"),
        "vae": os.environ.get("VAE_NAME", "qwen_image_2.1_vae_bf16.safetensors"),
        "comfy_ready": False,
    }
    payload["quant"] = os.environ.get("QUANT_LABEL", "—")
    te = payload["text_encoder"]
    payload["text_encoder_short"] = "int8" if "int8" in te else "bf16"

    try:
        stats = comfy_get("/system_stats")
        payload["comfy_ready"] = True
        for dev in stats.get("devices", []):
            if dev.get("type") == "cuda":
                payload["gpu_name"] = dev.get("name")
                payload["vram_total_gb"] = (dev.get("vram_total") or 0) / (1024 ** 3)
                payload["vram_free_gb"] = (dev.get("vram_free") or 0) / (1024 ** 3)
                break
    except Exception as exc:  # noqa: BLE001
        payload["comfy_error"] = f"{type(exc).__name__}: {exc}"

    return JSONResponse(payload)


@app.post("/api/generate")
async def api_generate(request: Request) -> JSONResponse:
    try:
        req = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    if not (req.get("prompt") or "").strip():
        return JSONResponse({"error": "prompt 不能为空"}, status_code=400)

    try:
        info = object_info()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    # ComfyUI ships a dedicated Qwen-Image-2.1 text-encoder node, but its
    # schema has changed between versions. Try it first, then fall back to the
    # generic CLIPTextEncode graph rather than failing the whole request.
    errors: list[str] = []
    res = None
    for prefer_modern in (True, False):
        try:
            wf = build_workflow(req, info, prefer_modern=prefer_modern)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"构建工作流失败: {type(exc).__name__}: {exc}")
            break

        r = http.post(
            f"{COMFY_URL}/prompt", json={"prompt": wf, "client_id": "qwen21-webui"}
        )
        if r.status_code == 200:
            res = r.json()
            break

        body = r.text[:1500]
        label = "TextEncodeQwenImage21" if prefer_modern else "CLIPTextEncode"
        errors.append(f"[{label}] HTTP {r.status_code} {body}")
        if not prefer_modern or "TextEncodeQwenImage21" not in body:
            break

    if res is None:
        return JSONResponse(
            {"error": "ComfyUI 拒绝了工作流: " + " || ".join(errors)}, status_code=502
        )

    pid = res.get("prompt_id")
    with PROGRESS_LOCK:
        PROGRESS[pid] = {"progress": 2.0, "step": 0, "total": int(req.get("steps", 25))}
    return JSONResponse({"prompt_id": pid, "number": res.get("number")})


@app.get("/api/result/{prompt_id}")
def api_result(prompt_id: str) -> JSONResponse:
    with PROGRESS_LOCK:
        prog = dict(PROGRESS.get(prompt_id, {}))

    try:
        hist = comfy_get(f"/history/{prompt_id}")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    if prompt_id in hist:
        entry = hist[prompt_id]
        status = entry.get("status", {})
        images = []
        for node_out in (entry.get("outputs") or {}).values():
            for img in node_out.get("images", []) or []:
                qs = urllib.parse.urlencode(
                    {
                        "filename": img.get("filename", ""),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    }
                )
                images.append({"filename": img.get("filename"), "url": f"/api/image?{qs}"})

        if status.get("status_str") == "error" or not status.get("completed", True):
            msgs = status.get("messages") or []
            return JSONResponse({"status": "error", "error": json.dumps(msgs, ensure_ascii=False)[:800]})

        return JSONResponse({"status": "done", "progress": 100, "images": images})

    # still running?
    try:
        q = comfy_get("/queue")
        running_ids = [item[1] for item in (q.get("queue_running") or [])]
        pending_ids = [item[1] for item in (q.get("queue_pending") or [])]
    except Exception:  # noqa: BLE001
        running_ids, pending_ids = [], []

    if prompt_id in running_ids:
        return JSONResponse({"status": "running", **prog})
    if prompt_id in pending_ids or not running_ids:
        return JSONResponse({"status": "queued", "progress": prog.get("progress", 1.0)})
    return JSONResponse({"status": "running", **prog})


@app.post("/api/interrupt")
def api_interrupt() -> JSONResponse:
    try:
        comfy_post("/interrupt", {})
    except Exception:  # noqa: BLE001
        pass
    try:
        comfy_post("/queue", {"clear": True})
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse({"ok": True})


@app.get("/api/image")
def api_image(filename: str, subfolder: str = "", type: str = "output") -> Response:
    r = http.get(f"{COMFY_URL}/view", params={"filename": filename, "subfolder": subfolder, "type": type})
    r.raise_for_status()
    return Response(
        r.content,
        media_type=r.headers.get("content-type", "image/png"),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/upload")
async def api_upload(request: Request, filename: str = "") -> JSONResponse:
    """Accept a reference image and drop it into ComfyUI's input directory."""
    data = await request.body()
    if not data:
        return JSONResponse({"error": "空文件"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "文件过大（上限 32 MB）"}, status_code=413)

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(filename or ""))
    if not safe:
        safe = "upload.png"
    if not Path(safe).suffix:
        safe += ".png"

    # content hash makes re-uploads idempotent and stops files clobbering
    name = f"{hashlib.md5(data).hexdigest()[:8]}_{safe}"

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / name).write_bytes(data)

    return JSONResponse(
        {"name": name, "url": "/api/input_image?filename=" + urllib.parse.quote(name)}
    )


@app.get("/api/input_image")
def api_input_image(filename: str) -> Response:
    r = http.get(f"{COMFY_URL}/view", params={"filename": filename, "type": "input"})
    r.raise_for_status()
    return Response(
        r.content,
        media_type=r.headers.get("content-type", "image/png"),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/history")
def api_history(limit: int = 12) -> JSONResponse:
    try:
        hist = comfy_get("/history", max_items=limit)
    except Exception:  # noqa: BLE001
        return JSONResponse({"items": []})

    items = []
    for pid, entry in list(hist.items())[:limit]:
        prompt_block = (entry.get("prompt") or [None, {}, {}])
        meta = prompt_block[2] if len(prompt_block) > 2 else {}
        for node_id, node_out in (entry.get("outputs") or {}).items():
            for img in node_out.get("images", []) or []:
                qs = urllib.parse.urlencode(
                    {
                        "filename": img.get("filename", ""),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    }
                )
                items.append(
                    {
                        "filename": img.get("filename"),
                        "url": f"/api/image?{qs}",
                        "seed": meta.get("seed"),
                        "steps": meta.get("steps"),
                        "width": meta.get("width"),
                        "height": meta.get("height"),
                    }
                )
    return JSONResponse({"items": items})


# --------------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    fetch_assets()
    threading.Thread(target=ws_progress_listener, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
