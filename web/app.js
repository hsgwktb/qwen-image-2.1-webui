/* ==========================================================================
   Qwen-Image-2.1 Uncensored — WebUI front-end
   Fetched from the GitHub repository at runtime by server.py.
   ========================================================================== */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    status: $("dot-status"),
    statusText: $("txt-status"),
    vram: $("txt-vram"),
    quant: $("txt-quant"),
    te: $("txt-te"),
    banner: $("banner"),
    bannerText: $("banner-text"),

    prompt: $("prompt"),
    promptLen: $("prompt-len"),
    negative: $("negative"),

    drop: $("drop"),
    file: $("file"),
    refs: $("refs"),
    editTag: $("edit-tag"),
    resolution: $("resolution"),
    resolutionVal: $("resolution-val"),

    presets: $("presets"),
    width: $("width"),
    height: $("height"),
    batch: $("batch"),
    batchVal: $("batch-val"),

    steps: $("steps"),
    stepsVal: $("steps-val"),
    cfg: $("cfg"),
    cfgVal: $("cfg-val"),
    sampler: $("sampler"),
    scheduler: $("scheduler"),
    seed: $("seed"),
    seedVal: $("seed-val"),
    btnRand: $("btn-rand"),

    btnGen: $("btn-gen"),
    btnGenLabel: $("btn-gen-label"),
    btnStop: $("btn-stop"),
    bar: $("bar"),
    progLeft: $("prog-left"),
    progRight: $("prog-right"),

    gallery: $("gallery"),
    empty: $("empty"),
    count: $("chip-count"),
    btnClear: $("btn-clear"),

    log: $("log"),
    assetSrc: $("asset-src"),
  };

  const PRESETS = [
    { label: "1:1 1MP", w: 1024, h: 1024 },
    { label: "1:1 2K", w: 2048, h: 2048 },
    { label: "4:3", w: 1152, h: 896 },
    { label: "3:4", w: 896, h: 1152 },
    { label: "16:9", w: 1344, h: 768 },
    { label: "9:16", w: 768, h: 1344 },
  ];

  const DEFAULT_NEG =
    "低分辨率, 低质量, 最差质量, 模糊, 变形, 畸形, 多余的手指, 文字, 水印, 签名";

  let cfg = {};
  let polling = null;
  let inflight = 0;

  // reference images for image editing; index 0 is the edit target
  let REFS = [];
  const MAX_REFS = 10;

  /* ------------------------------ logging ------------------------------ */

  function nowStamp() {
    const d = new Date();
    return d.toTimeString().slice(0, 8);
  }

  function log(msg, kind = "i") {
    const line = document.createElement("div");
    line.innerHTML =
      '<span class="t">' + nowStamp() + "</span> " +
      '<span class="' + kind + '"></span>';
    line.lastChild.textContent = msg;
    els.log.appendChild(line);
    els.log.scrollTop = els.log.scrollHeight;
  }

  /* ------------------------------ helpers ------------------------------ */

  function setStatus(state, text) {
    els.status.className = "dot" + (state ? " " + state : "");
    els.statusText.textContent = text;
  }

  function showBanner(text) {
    if (!text) {
      els.banner.classList.remove("show");
      return;
    }
    els.bannerText.textContent = text;
    els.banner.classList.add("show");
  }

  function setProgress(pct, left, right) {
    els.bar.style.width = Math.max(0, Math.min(100, pct)) + "%";
    if (left !== undefined) els.progLeft.textContent = left;
    if (right !== undefined) els.progRight.textContent = right;
  }

  function setBusy(busy) {
    els.btnGen.disabled = busy;
    els.btnStop.disabled = !busy;
    els.btnGen.classList.toggle("loading", busy);
    els.btnGenLabel.textContent = busy ? "生成中…" : "生成";
  }

  function updateCount() {
    const n = els.gallery.querySelectorAll("img").length;
    els.count.textContent = n + " 张";
  }

  /* ------------------------------ config ------------------------------- */

  async function loadConfig() {
    try {
      const r = await fetch("/api/config");
      cfg = await r.json();
    } catch (e) {
      showBanner("无法连接后端：" + e.message);
      setStatus("err", "离线");
      return;
    }

    els.quant.textContent = cfg.quant || "—";
    els.te.textContent = cfg.text_encoder_short || "—";
    els.assetSrc.textContent = cfg.asset_source || "—";

    if (els.editTag) {
      const ok = !!cfg.editing_available;
      els.editTag.textContent = ok ? "可编辑" : "不可用";
      els.editTag.className = "tag " + (ok ? "on" : "off");
      els.editTag.title = ok
        ? "ComfyUI 提供 TextEncodeQwenImage21，支持参考图编辑"
        : "当前 ComfyUI 缺少 TextEncodeQwenImage21 节点，只能文生图";
    }

    if (cfg.max_resolution && els.resolution) {
      els.resolution.max = String(cfg.max_resolution);
    }
    log(
      cfg.downscale_on_upload
        ? "上传图片会等比缩放到长边 " + cfg.max_upload_side + " px。"
        : "上传图片保持原始分辨率，不做缩放。",
      "i",
    );

    if (cfg.gpu_name) {
      els.vram.textContent =
        (cfg.vram_free_gb != null ? cfg.vram_free_gb.toFixed(1) : "?") + " / " +
        (cfg.vram_total_gb != null ? cfg.vram_total_gb.toFixed(1) : "?") + " GB";
    }

    if (!cfg.comfy_ready) {
      setStatus("busy", "后端启动中…");
      showBanner("ComfyUI 尚未就绪，模型可能仍在下载或加载。" + (cfg.comfy_error ? " " + cfg.comfy_error : ""));
    } else {
      setStatus("ok", "就绪");
      showBanner("");
    }

    // seed display
    if (Number(els.seed.value) < 0) els.seedVal.textContent = "随机";
    else els.seedVal.textContent = els.seed.value;
  }

  /* ------------------------------ gallery ------------------------------ */

  function addCard(img, meta) {
    if (els.empty) els.empty.style.display = "none";

    const card = document.createElement("div");
    card.className = "card";

    const image = document.createElement("img");
    image.src = img.url;
    image.loading = "lazy";
    image.alt = "generated";
    image.addEventListener("click", () => window.open(img.url, "_blank"));
    image.style.cursor = "zoom-in";

    const bar = document.createElement("div");
    bar.className = "meta";

    const seed = document.createElement("span");
    seed.className = "seed";
    seed.textContent = "seed " + meta.seed + " · " + meta.width + "×" + meta.height + " · " + meta.steps + "步";

    const acts = document.createElement("span");
    acts.className = "acts";

    const a = document.createElement("a");
    a.href = img.url;
    a.download = img.filename || "qwen-image.png";
    a.textContent = "下载";

    const reuse = document.createElement("button");
    reuse.type = "button";
    reuse.textContent = "复用种子";
    reuse.addEventListener("click", () => {
      els.seed.value = meta.seed;
      els.seedVal.textContent = meta.seed;
    });

    acts.append(a, reuse);
    bar.append(seed, acts);
    card.append(image, bar);
    els.gallery.prepend(card);
    updateCount();
  }

  function addPending(n) {
    const holders = [];
    for (let i = 0; i < n; i++) {
      const d = document.createElement("div");
      d.className = "card pending";
      d.textContent = "排队中…";
      els.gallery.prepend(d);
      holders.push(d);
    }
    if (els.empty) els.empty.style.display = "none";
    return holders;
  }

  /* --------------------------- reference images ------------------------ */

  function renderRefs() {
    els.refs.innerHTML = "";
    REFS.forEach((r, i) => {
      const card = document.createElement("div");
      card.className = "ref";

      const img = document.createElement("img");
      img.src = r.url;
      img.alt = r.name;
      img.title = r.name;

      const idx = document.createElement("span");
      idx.className = "idx" + (i === 0 ? " target" : "");
      idx.textContent = "image" + (i + 1);

      const dim = document.createElement("span");
      dim.className = "dim" + (r.resized ? " shrunk" : "");
      if (r.width && r.height) {
        dim.textContent = r.width + "×" + r.height;
        dim.title = r.resized
          ? "原图 " + r.originalWidth + "×" + r.originalHeight + "，已等比缩放"
          : "原始分辨率，未缩放";
      }

      const del = document.createElement("button");
      del.type = "button";
      del.className = "del";
      del.textContent = "×";
      del.title = "移除";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        REFS.splice(i, 1);
        renderRefs();
      });

      card.append(img, idx, dim, del);
      els.refs.appendChild(card);
    });
  }

  async function addFiles(files) {
    const list = [...files].filter((f) => f.type.startsWith("image/"));
    if (!list.length) return;

    for (const f of list) {
      if (REFS.length >= MAX_REFS) {
        showBanner("最多 " + MAX_REFS + " 张参考图，多余的已忽略。");
        break;
      }
      try {
        const r = await fetch("/api/upload?filename=" + encodeURIComponent(f.name), {
          method: "POST",
          headers: { "Content-Type": f.type || "application/octet-stream" },
          body: f,
        });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || "HTTP " + r.status);
        REFS.push({
          name: j.name,
          url: j.url,
          width: j.width,
          height: j.height,
          originalWidth: j.original_width,
          originalHeight: j.original_height,
          resized: !!j.resized,
        });
        log(
          j.message ? "已上传 " + j.name + " · " + j.message : "已上传参考图 " + j.name,
          j.resized ? "w" : "i",
        );
      } catch (e) {
        log("上传失败 " + f.name + "：" + e.message, "e");
        showBanner("上传失败：" + e.message);
      }
    }
    renderRefs();
  }

  function wireUploader() {
    els.drop.addEventListener("click", () => els.file.click());
    els.file.addEventListener("change", () => {
      addFiles(els.file.files);
      els.file.value = "";
    });

    ["dragenter", "dragover"].forEach((ev) =>
      els.drop.addEventListener(ev, (e) => {
        e.preventDefault();
        els.drop.classList.add("over");
      }),
    );
    ["dragleave", "drop"].forEach((ev) =>
      els.drop.addEventListener(ev, (e) => {
        e.preventDefault();
        els.drop.classList.remove("over");
      }),
    );
    els.drop.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
    });
  }

  /* ------------------------------ generate ----------------------------- */

  async function generate() {
    const prompt = els.prompt.value.trim();
    if (!prompt) {
      showBanner("请先填写提示词。");
      return;
    }
    showBanner("");

    let seed = Number(els.seed.value);
    if (!Number.isFinite(seed) || seed < 0) seed = Math.floor(Math.random() * 1e15);

    const payload = {
      prompt,
      negative: els.negative.value.trim() || DEFAULT_NEG,
      width: Number(els.width.value) || 1024,
      height: Number(els.height.value) || 1024,
      steps: Number(els.steps.value) || 25,
      cfg: Number(els.cfg.value),
      sampler: els.sampler.value,
      scheduler: els.scheduler.value,
      seed,
      batch: Number(els.batch.value) || 1,
      images: REFS.map((r) => r.name),
      resolution: Number(els.resolution.value),
    };

    setBusy(true);
    setProgress(3, "已提交", "");
    const holders = addPending(payload.batch);
    log(
      (payload.images.length ? "编辑" : "文生图") +
        " · " + payload.width + "×" + payload.height +
        " · " + payload.steps + "步 · seed " + seed +
        (payload.images.length ? " · " + payload.images.length + " 张参考图" : ""),
      "i",
    );

    let res;
    try {
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      res = await r.json();
      if (!r.ok) throw new Error(res.error || ("HTTP " + r.status));
    } catch (e) {
      holders.forEach((h) => h.remove());
      setBusy(false);
      setProgress(0, "失败", "");
      log("提交失败: " + e.message, "e");
      showBanner("提交失败：" + e.message);
      return;
    }

    pollResult(res.prompt_id, payload, holders);
  }

  function pollResult(promptId, payload, holders) {
    clearInterval(polling);
    const startedAt = Date.now();

    polling = setInterval(async () => {
      let st;
      try {
        const r = await fetch("/api/result/" + encodeURIComponent(promptId));
        st = await r.json();
      } catch (e) {
        return; // transient; keep polling
      }

      if (st.progress != null) {
        setProgress(
          st.progress,
          st.status === "running" ? "采样中" : st.status,
          (st.step ? st.step + "/" + st.total : "") + (st.node ? "  " + st.node : "")
        );
      }

      if (st.status === "done") {
        clearInterval(polling);
        polling = null;
        holders.forEach((h) => h.remove());

        const images = st.images || [];
        if (!images.length) {
          log("完成，但没有返回图像。", "w");
        }
        images.forEach((img) => addCard(img, payload));

        const secs = ((Date.now() - startedAt) / 1000).toFixed(1);
        setBusy(false);
        setProgress(100, "完成", secs + "s");
        log("完成，用时 " + secs + "s，输出 " + images.length + " 张", "s");
        inflight = Math.max(0, inflight - 1);
        return;
      }

      if (st.status === "error") {
        clearInterval(polling);
        polling = null;
        holders.forEach((h) => h.remove());
        setBusy(false);
        setProgress(0, "出错", "");
        log("错误: " + (st.error || "未知"), "e");
        showBanner("生成失败：" + (st.error || "未知错误"));
        return;
      }
    }, 900);
  }

  async function interrupt() {
    try {
      await fetch("/api/interrupt", { method: "POST" });
      log("已发送中断请求", "w");
    } catch (e) {
      log("中断失败: " + e.message, "e");
    }
  }

  /* ------------------------------ wiring ------------------------------- */

  function buildPresets() {
    PRESETS.forEach((p) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "preset";
      b.textContent = p.label;
      b.addEventListener("click", () => {
        els.width.value = p.w;
        els.height.value = p.h;
        [...els.presets.children].forEach((c) => c.classList.remove("active"));
        b.classList.add("active");
      });
      els.presets.appendChild(b);
    });
  }

  function wire() {
    wireUploader();

    els.prompt.addEventListener("input", () => {
      els.promptLen.textContent = els.prompt.value.length;
    });

    els.batch.addEventListener("input", () => {
      els.batchVal.textContent = els.batch.value;
    });

    els.resolution.addEventListener("input", () => {
      const v = Number(els.resolution.value);
      els.resolutionVal.textContent = v === 0 ? "原始尺寸" : v;
    });
    els.steps.addEventListener("input", () => {
      els.stepsVal.textContent = els.steps.value;
    });
    els.cfg.addEventListener("input", () => {
      els.cfgVal.textContent = Number(els.cfg.value).toFixed(1);
    });
    els.seed.addEventListener("input", () => {
      els.seedVal.textContent = Number(els.seed.value) < 0 ? "随机" : els.seed.value;
    });
    els.btnRand.addEventListener("click", () => {
      els.seed.value = -1;
      els.seedVal.textContent = "随机";
    });

    els.btnGen.addEventListener("click", generate);
    els.btnStop.addEventListener("click", interrupt);

    els.btnClear.addEventListener("click", () => {
      els.gallery.querySelectorAll(".card").forEach((c) => c.remove());
      if (els.empty) els.empty.style.display = "";
      updateCount();
    });

    els.prompt.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        generate();
      }
    });
  }

  async function loadHistory() {
    try {
      const r = await fetch("/api/history?limit=12");
      const h = await r.json();
      (h.items || []).reverse().forEach((it) => addCard(it, it));
    } catch (_) {
      /* history is best-effort */
    }
  }

  async function init() {
    buildPresets();
    wire();
    log("WebUI 已加载，前端资源来自 GitHub。", "s");
    await loadConfig();
    await loadHistory();
    setInterval(loadConfig, 15000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
