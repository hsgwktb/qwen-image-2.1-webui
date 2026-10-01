# Qwen-Image-2.1 Uncensored — Colab WebUI

在 Google Colab 的 **L4 GPU** 上部署 [`0xSojalSec/Qwen-Image-2.1-Uncensored-HF`](https://huggingface.co/0xSojalSec/Qwen-Image-2.1-Uncensored-HF)
的 GGUF 量化模型，并启动一个自定义 WebUI。

前端文件（`index.html` / `style.css` / `app.js`）保存在本仓库中，运行时由服务端拉取，
**没有使用 base64 内联**。

---

## 快速开始

1. 用 Colab 打开本仓库的 notebook：

   ```
   https://colab.research.google.com/github/hsgwktb/qwen-image-2.1-webui/blob/main/qwen_image_2.1_webui.ipynb
   ```

2. `运行时` → `更改运行时类型` → 硬件加速器选 **L4 GPU**。
3. `全部运行`。首次约 8–15 分钟（主要是下载 ~17 GB 权重）。
4. 最后一个单元会打印 WebUI 地址。

---

## 显存与量化的选择

脚本读取实际显存后自动选型：

| 显存 | DiT 量化 | 权重 | 文本编码器 |
| --- | --- | --- | --- |
| ≥ 40 GB | BF16 | 14.23 GB | bf16 (17.5 GB) |
| **20–40 GB（L4）** | **Q8_0** | **7.59 GB** | **int8 (9.35 GB)** |
| 14–20 GB | Q6_K | 5.88 GB | int8 |
| 10–14 GB | Q5_K_M | 5.22 GB | int8 |
| < 10 GB | Q4_K_M | 4.60 GB | int8 |

**L4（24 GB）上用 Q8_0。** 上游作者推荐的 Q4_K_M 面向 8–12 GB 消费级显卡；
在 24 GB 显存下，Q8_0（7.59 GB）+ int8 文本编码器（9.35 GB）+ VAE（0.68 GB）合计约 17.6 GB，
留出约 6 GB 给采样激活值，没有必要牺牲画质。

也可以手动指定：

```python
os.environ["QWEN_QUANT"] = "Q6_K"
```

---

## 图像编辑

Qwen-Image-2.1 的编辑能力用的是**同一套权重** —— 不需要额外下载模型，只是把参考图喂给同一个
`TextEncodeQwenImage21` 节点。WebUI 左侧「参考图 · 图像编辑」面板就是入口。

用法：

1. 拖拽或点击上传参考图（**最多 10 张**）
2. **第 1 张是编辑目标**，其余是参考图
3. 在提示词里用 `<image1>`、`<image2>` 引用它们，例如
   `把 <image1> 的衣服换成 <image2> 的款式`
4. 点生成。**留空参考图就自动回到纯文生图**

几条来自官方模板的注意事项：

- **输出尺寸跟随 image_1**；其他参考图可以尺寸/比例不同
- `resolution` 是**总像素预算**（不是宽高），默认 1024，模型原生支持到 2048
- 官方推荐 euler + 40–50 步；cfg 保持 1，此时 negative prompt 不生效
- 编辑时画布取自 image_1 的 latent（`KSampler.latent_image` 接编码节点的第三个输出），
  而不是空白 latent

### 上传不缩放

参考图**按原样保存**：`/api/upload` 不做任何缩放或重编码，写进 ComfyUI `input/` 的就是原始
字节，分辨率与原图完全一致（PNG 的 alpha 通道自然也保留）。6000×4000 进来的还是 6000×4000。

如果确实想限制上传尺寸，把 `MAX_UPLOAD_SIDE` 设成非 0（例如 2048），服务端才会等比缩放，
且只处理超过阈值的长边，未超过的仍原样保留。**默认是 0 = 关闭缩放。**

> `resolution` 滑块控制的是**模型再采样**的尺寸，默认 **0 = 保持上传时的原始分辨率**。
> 设成非 0 时模型会把参考图缩到约 `resolution × resolution`（保持比例，模型原生支持到 2048）——
> 参考图很大时这会明显更快、更省显存。

参考图**不是**合并成一个张量再喂进去的。`TextEncodeQwenImage21` 的 `images` 是
**Autogrow 输入**（实机 `/object_info` 里类型为 `COMFY_AUTOGROW_V3`）：节点收到的是一份以
`image_1`…`image_16` 为键的字典，所以工作流里必须写成点号键 `images.image_1`、`images.image_2`…
传批处理张量会让节点内部的 `images or {}` 抛
`Boolean value of Tensor with more than one value is ambiguous`。

上游还有一个**可选的提示词增强器**（`qwen3.5_9b_qwen_image_2.1_pe_i2i`，8.82 GB），
用 Qwen3.5 把「换衣服」这类短指令改写成完整描述再交给编辑模型，效果更好但要额外下载权重，
本仓库没有启用。

---

## 架构

```
Colab L4 runtime
├── ComfyUI  (:8188)          ← 推理后端，UnetLoaderGGUF 加载 DiT
│   ├── models/diffusion_models/qwen-image-2.1-UC-Q8_0.gguf
│   ├── models/text_encoders/qwen3vl_8b_int8_convrot.safetensors
│   └── models/vae/qwen_image_2.1_vae_bf16.safetensors
└── server.py (:7860)          ← 自定义 WebUI
    ├── GET  /                 ← 由 GitHub 拉取的 index.html
    ├── GET  /assets/*         ← 由 GitHub 拉取的 style.css / app.js
    ├── POST /api/generate     ← 构造工作流并提交给 ComfyUI（带 images 即为编辑）
    ├── GET  /api/result/{id}  ← 轮询进度与结果
    ├── POST /api/upload       ← 参考图写入 ComfyUI 的 input/ 目录
    ├── GET  /api/input_image  ← 代理 ComfyUI 的 /view?type=input（参考图预览）
    ├── GET  /api/image        ← 代理 ComfyUI 的 /view
    └── GET  /api/history      ← 最近生成记录
```

进度条通过 ComfyUI 的 WebSocket（`/ws`）实时获取采样步数。

---

## 前端资源的两种加载方式

`server.py` 用 `ASSET_SOURCE` 控制：

| 值 | 行为 |
| --- | --- |
| `proxy`（默认） | 服务端在 Colab 上从 GitHub 拉取文件，再以同源 `/assets/*` 提供给浏览器。国内网络也能用。 |
| `github` | 页面直接 `<link>` / `<script>` 指向 GitHub raw（失败时回退 jsDelivr）。 |

两种方式下网页文件的唯一来源都是 GitHub，notebook 里不含任何 Web 代码。

```python
os.environ["ASSET_SOURCE"] = "github"
```

---

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `qwen_image_2.1_webui.ipynb` | Colab 入口 notebook（很薄，逻辑都在 GitHub 上） |
| `colab_setup.py` | 部署脚本：装依赖 → 下载权重 → 起 ComfyUI → 起 WebUI |
| `server.py` | WebUI 服务端：拉取前端资源 + 驱动 ComfyUI API |
| `web/index.html` | 页面结构 |
| `web/style.css` | 样式 |
| `web/app.js` | 前端逻辑 |

---

## 排错

| 现象 | 处理 |
| --- | --- |
| `Unknown model architecture!` | ComfyUI-GGUF 用 `leejet` fork（脚本已默认）；`city96` 版本不支持 Qwen-Image 2.1 |
| CUDA out of memory | 降低量化：`QWEN_QUANT=Q5_K_M`，或让 `start_comfyui(lowvram=True)` |
| WebUI 打不开 | 确认已登录同一 Google 账号；查 `/content/webui.log` |
| ComfyUI 起不来 | 查 `/content/comfyui.log` |
| 下载中断 | 重新运行同一个单元，`huggingface_hub` 会断点续传 |

---

## 许可与内容说明

- 模型遵循 **Qwen Research License**，量化权重来自 `0xSojalSec/Qwen-Image-2.1-Uncensored-HF`。
- 该模型**没有内置安全过滤器**，输出完全取决于提示词。请确认在你所在地区合法合规，并自行承担使用责任。
