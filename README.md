# Marimo AI - High-Performance Qwen3.8-27B FastMTP Server

A reactive [Marimo](https://marimo.io) notebook designed for cloud environments like **molab** (NVIDIA RTX PRO 6000 Blackwell with 98 GB VRAM).

It compiles a custom-patched `llama-server` supporting **HauhauCS FastMTP** (speculative multi-token prediction) and runs the **Qwen3.8-27B Uncensored Aggressive (`Q8_K_P`)** model entirely in GPU memory with native vision (`mmproj-BF16`) and an OpenAI-compatible API endpoint for agents like **Hermes**.

## 🚀 One-Click Launch in molab

Open this notebook directly in molab:
- **URL:** [Open in molab](https://molab.marimo.io/?notebook=https://github.com/ramin-mahmoodi/marimo-ai/blob/main/qwen38_fastmtp_molab.py)

## 📦 What's Included
- `qwen38_fastmtp_molab.py`: Full standalone Marimo notebook.
- Native Blackwell CUDA build (`sm_120`) with Ninja and CCCL integration.
- High-speed Rust-based downloads using `hf_transfer`.
- 100% GPU VRAM offload (`--no-mmap`) protecting system RAM.
- Secure TCP 443 tunneling via `zrok`.
