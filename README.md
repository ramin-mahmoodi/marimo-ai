# 🚀 Marimo AI - High-Performance Qwen3.8-27B FastMTP Server

[![Open in molab](https://marimo.io/shield.svg)](https://molab.marimo.io/?notebook=https://github.com/ramin-mahmoodi/marimo-ai/blob/main/qwen38_fastmtp_molab.py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![NVIDIA Blackwell](https://img.shields.io/badge/GPU-RTX_PRO_6000_Blackwell_(98GB)-76B900.svg)](https://www.nvidia.com)
[![Model](https://img.shields.io/badge/Model-Qwen3.8--27B--Uncensored--Q8_K_P-orange.svg)](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF)

A production-grade, reactive [Marimo](https://marimo.io) notebook engineered for cloud GPU platforms (specifically **molab** with an **NVIDIA RTX PRO 6000 Blackwell Server Edition, 98 GB VRAM**).

This server hosts **Qwen3.8-27B-Uncensored-Aggressive** at high-precision **`Q8_K_P`** (31.46 GB) with the custom **HauhauCS FastMTP** speculative decoding acceleration profile, native multimodal vision (`mmproj-BF16`), and an OpenAI-compliant API endpoint tailored for autonomous AI agents like **Hermes**.

---

## ⚡ Live Performance Benchmarks (Verified on RTX PRO 6000 Blackwell)

The following metrics were measured live during continuous operation under realistic workloads:

| Metric | Measured Value | Details |
| :--- | :--- | :--- |
| **Prompt Processing (PP / Prefill)** | **`3,262.11` tok/s** | Processed **15,391 tokens in 5.47s** (`0.36 ms/tok`) |
| **Token Generation (TG / Decode)** | **`55.5 – 60.2` tok/s** | High-precision 8-bit (`Q8_K_P`) dense 27B model |
| **Time To First Token (TTFT)** | **`1.47` seconds** | Measured end-to-end over the public `zrok` HTTPS tunnel |
| **Speculative Draft Acceptance** | **`32.4%`** | 3-token speculative depth via FastMTP sidecar |
| **Configured Context Window** | **`131,072` tokens** | Native FlashAttention offloaded 100% to VRAM |
| **System RAM Footprint** | **`< 10 GiB` / 160 GiB** | Protected by `--no-mmap` to eliminate OOM risks |
| **GPU VRAM Usage** | **`~42 GiB` / 98 GiB** | Leaves 56 GiB headroom for ultra-long context |

---

## 🛠️ Key Architectural Highlights

1. **Precompiled FastMTP Engine (Zero Build Wait):**
   * Uses a custom build of `llama-server` compiled specifically for NVIDIA Blackwell (`sm_120`) with CUDA 12.8, Ninja, and CCCL headers.
   * Auto-downloads precompiled binaries directly from this repository's [Releases](https://github.com/ramin-mahmoodi/marimo-ai/releases) in **3 seconds**—completely eliminating the 10-minute compile step.
2. **High-Speed Rust `hf_transfer` Streaming:**
   * Uses Rust-accelerated multi-threaded downloads (`hf-transfer`) to pull the full 31.46 GB model directly to NVMe storage in **~60 to 90 seconds** at 300–500 MB/s.
3. **Hermes Agent Optimization:**
   * **`--jinja`**: Full support for native Tool Calling & Function Calling.
   * **`--reasoning-format deepseek`**: Separates internal `<think>` tokens into `reasoning_content` so JSON tool calls never break parsing.
   * **`--reasoning-preserve`**: Retains multi-turn reasoning chains across agent execution steps.
4. **Firewall-Proof Egress via zrok:**
   * Utilizes OpenZiti's `zrok` proxy over standard TCP port 443 (HTTPS), bypassing restrictive container firewalls that block Cloudflare Quick Tunnels (UDP 7844).

---

## 🚀 Quickstart Guide

### 1. Launch in molab
Click the badge above or navigate to:
```text
https://molab.marimo.io/?notebook=https://github.com/ramin-mahmoodi/marimo-ai/blob/main/qwen38_fastmtp_molab.py
```

### 2. Configure & Build (Cell 1 & 2)
1. Select your desired Context Size (default: `131,072`).
2. Keep **FastMTP Depth 3** and **Reasoning Effort `xhigh`**.
3. Click **`1. Build Patched llama.cpp + Download Q8_K_P & FastMTP`**.
   *(If the precompiled release asset is available, setup takes under 2 minutes).*

### 3. Start Public Tunnel (Cell 3 & 4)
1. Get a free token from [zrok.io](https://zrok.io) (email only, no credit card).
2. Paste the token into the `zrok account token` field.
3. Click **`2. Launch / Restart GPU Server & Public Tunnel`**.

---

## 🔗 Connecting to Hermes & OpenAI Clients

Once online, the notebook displays your live connection credentials:

| Parameter | Value |
| :--- | :--- |
| **OpenAI Base URL** | `https://<unique-id>.shares.zrok.io/v1` |
| **Chat Completions URL** | `https://<unique-id>.shares.zrok.io/v1/chat/completions` |
| **Model Name** | `qwen3.8-27b` |
| **API Key** | `sk-molab-local` *(any non-empty string)* |

### Example cURL Request:
```bash
curl https://<unique-id>.shares.zrok.io/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-molab-local" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Hello! Explain your architecture."}
    ],
    "temperature": 0.7
  }'
```

### Python SDK Example:
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<unique-id>.shares.zrok.io/v1",
    api_key="sk-molab-local",
)

response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Analyze this code..."}],
)
print(response.choices[0].message.content)
```

---

## 📁 Repository Structure
```text
├── .github/workflows/
│   └── build_llama_server.yml   # Automated GitHub Actions CUDA 12.8 Blackwell builder
├── qwen38_fastmtp_molab.py      # Standalone reactive Marimo notebook
└── README.md                    # Architecture, setup, and live benchmarks
```

## 📜 License
- Base Model: [Qwen 3.8-27B License](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0).
- Quantization & FastMTP Profile: Developed by [HauhauCS](https://huggingface.co/HauhauCS).
- Notebook & Deployment Pipeline: Licensed under Apache 2.0.
