# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cmake==4.4.4",
#     "hf-transfer",
#     "huggingface-hub",
#     "marimo",
#     "ninja==1.13.2",
#     "requests",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import time
    import json
    import shutil
    import signal
    import socket
    import re
    import subprocess
    from pathlib import Path
    import requests

    # Persistent singleton state tied to sys module so Marimo reactive re-runs
    # never lose track of running background processes or spawn duplicates.
    if not hasattr(sys, "_qwen_molab_state"):
        sys._qwen_molab_state = {
            "llama_proc": None,
            "cf_proc": None,
            "public_url": None,
            "work_dir": None,
        }
    state_store = sys._qwen_molab_state

    return (
        Path,
        mo,
        os,
        re,
        requests,
        shutil,
        state_store,
        subprocess,
        sys,
        time,
    )


@app.cell
def _(Path, mo, shutil, state_store, subprocess):
    def _pick_best_disk_dir() -> Path:
        """Selects a real disk path (avoiding RAM-backed tmpfs) with maximum free space."""
        _candidates = [
            Path.cwd() / "qwen38_workspace",
            Path("/workspace/qwen38_workspace"),
            Path.home() / "qwen38_workspace",
            Path("/var/tmp/qwen38_workspace"),
        ]
        _tmpfs_mounts = set()
        try:
            with open("/proc/mounts", "r", encoding="utf-8") as _f:
                for _line in _f:
                    _parts = _line.split()
                    if len(_parts) >= 3 and _parts[2] in ("tmpfs", "ramfs", "devtmpfs"):
                        _tmpfs_mounts.add(_parts[1])
        except Exception:
            pass

        _best_path = Path.cwd() / "qwen38_workspace"
        _best_free = -1

        for _cand in _candidates:
            try:
                _parent = _cand.parent
                if not _parent.exists():
                    continue
                _mount = "/"
                _resolved = str(_parent.resolve())
                try:
                    _df_out = subprocess.check_output(
                        ["df", "-P", _resolved], text=True
                    ).splitlines()
                    if len(_df_out) >= 2:
                        _mount = _df_out[1].split()[5]
                except Exception:
                    pass

                if _mount in _tmpfs_mounts:
                    continue

                _usage = shutil.disk_usage(_parent)
                if _usage.free > _best_free:
                    _best_free = _usage.free
                    _best_path = _cand
            except Exception:
                continue

        _best_path.mkdir(parents=True, exist_ok=True)
        return _best_path

    work_dir = _pick_best_disk_dir()
    state_store["work_dir"] = str(work_dir)

    _gpu_info = "GPU not detected"
    try:
        _gpu_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as _e:
        _gpu_info = f"nvidia-smi error: {_e}"

    _disk_free_gb = shutil.disk_usage(work_dir).free / (1024**3)

    # UI Controls
    ctx_slider = mo.ui.slider(
        start=32768,
        stop=204800,
        step=16384,
        value=131072,
        label="Context Size (--ctx-size)",
        show_value=True,
    )
    parallel_dropdown = mo.ui.dropdown(
        options={
            "2 Slots (Recommended for Multi-Chat & Concurrent Requests)": 2,
            "1 Slot (Single-Chat / Maximum FastMTP Speed)": 1,
            "4 Slots (High Concurrency / Multi-Agent)": 4,
        },
        value="2 Slots (Recommended for Multi-Chat & Concurrent Requests)",
        label="Parallel Slots (--parallel)",
    )
    depth_dropdown = mo.ui.dropdown(
        options={"Depth 3 (Recommended by HauhauCS for Blackwell)": 3, "Depth 2": 2},
        value="Depth 3 (Recommended by HauhauCS for Blackwell)",
        label="FastMTP Speculative Depth",
    )
    reasoning_dropdown = mo.ui.dropdown(
        options={
            "medium (Balanced - Recommended)": "medium",
            "low (Fast - Quick Responses)": "low",
            "xhigh (Deep Reasoning - Can be slow)": "xhigh",
        },
        value="medium (Balanced - Recommended)",
        label="Reasoning Effort",
    )
    enable_fastmtp_chk = mo.ui.checkbox(
        value=False,
        label="⚡ Enable FastMTP Speculative Decoding (Boosts single chat to ~60 tok/s; keep unchecked for rock-solid concurrent multi-chat)",
    )
    enable_vision_chk = mo.ui.checkbox(
        value=True, label="Load BF16 Vision Projector (mmproj - 931 MB)"
    )
    setup_btn = mo.ui.run_button(
        label="1. Build Patched llama.cpp + Download Q8_K_P & FastMTP",
        kind="success",
    )

    mo.md(
        f"""
        # 🚀 Qwen3.8-27B-Uncensored-Aggressive (`Q8_K_P` + `FastMTP`) Server for Hermes
        **Hardware Detected:** `{_gpu_info}`  
        **Workspace Path (Non-RAM Disk):** `{work_dir}` (**`{_disk_free_gb:.1f} GiB`** free)

        ---
        ### ⚙️ Server Configuration (100% GPU VRAM Offload + `--no-mmap`)
        {mo.hstack([ctx_slider, parallel_dropdown, reasoning_dropdown], justify="start", gap=2)}
        {mo.hstack([enable_fastmtp_chk, depth_dropdown], justify="start", gap=2)}
        {enable_vision_chk}

        {setup_btn}
        """
    )
    return (
        ctx_slider,
        depth_dropdown,
        enable_fastmtp_chk,
        enable_vision_chk,
        parallel_dropdown,
        reasoning_dropdown,
        setup_btn,
        work_dir,
    )


@app.cell
def _(
    Path,
    enable_vision_chk,
    mo,
    os,
    setup_btn,
    shutil,
    subprocess,
    sys,
    work_dir,
):
    start_server_btn = mo.ui.run_button(
        label="2. Launch / Restart GPU Server & Public Tunnel",
        kind="warn",
    )

    mo.stop(
        not setup_btn.value,
        mo.md(
            "ℹ️ *Click button **1. Build Patched llama.cpp + Download Q8_K_P & FastMTP** above to start installation and model download.*"
        ),
    )

    _HF_REPO = "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
    _HF_BASE = f"https://huggingface.co/{_HF_REPO}/resolve/main"
    _LLAMA_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"

    _MODEL_FILE = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
    _DRAFT_FILE = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"
    _MMPROJ_FILE = "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
    _PATCH_FILE = "HauhauCS-FastMTP-llama.cpp.patch"

    _models_dir = work_dir / "models"
    _models_dir.mkdir(parents=True, exist_ok=True)
    _llama_dir = work_dir / "llama.cpp"
    _llama_bin = _llama_dir / "build" / "bin" / "llama-server"

    _logs = []

    def _run_cmd(_cmd_list, _cwd=None, _env=None):
        _merged_env = os.environ.copy()
        _cuda_bin = "/usr/local/cuda/bin"
        if os.path.isdir(_cuda_bin) and _cuda_bin not in _merged_env.get("PATH", ""):
            _merged_env["PATH"] = f"{_cuda_bin}:{_merged_env.get('PATH', '')}"
            if os.path.exists(f"{_cuda_bin}/nvcc"):
                _merged_env.setdefault("CUDACXX", f"{_cuda_bin}/nvcc")
        if _env:
            _merged_env.update(_env)
        _p = subprocess.run(
            _cmd_list,
            cwd=_cwd,
            env=_merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if _p.returncode != 0:
            raise RuntimeError(
                f"Command failed ({' '.join(map(str, _cmd_list))}):\n{_p.stdout[-3000:]}"
            )
        return _p.stdout

    def _ensure_hf_accelerator():
        try:
            import hf_transfer
        except ImportError:
            _run_cmd([sys.executable, "-m", "pip", "install", "--quiet", "hf-transfer", "huggingface_hub"])
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    def _download_hf_model(_repo: str, _filename: str, _dest: Path, _min_bytes: int = 1024):
        """Ultra-fast multi-threaded download via Rust hf-transfer directly to disk."""
        if _dest.exists() and _dest.stat().st_size >= _min_bytes:
            _logs.append(
                f"✅ Already downloaded: `{_dest.name}` ({_dest.stat().st_size / (1024**3):.2f} GiB)"
            )
            return

        _ensure_hf_accelerator()
        from huggingface_hub import hf_hub_download
        
        _downloaded = hf_hub_download(
            repo_id=_repo,
            filename=_filename,
            local_dir=str(_dest.parent),
            local_dir_use_symlinks=False,
        )
        _logs.append(
            f"✅ Downloaded (HF-Transfer): `{_dest.name}` ({_dest.stat().st_size / (1024**3):.2f} GiB)"
        )

    def _download_file(_url: str, _dest: Path, _min_bytes: int = 1024):
        """Simple curl for small files (patch/binaries)."""
        if _dest.exists() and _dest.stat().st_size >= _min_bytes:
            _logs.append(f"✅ Already downloaded: `{_dest.name}`")
            return
        _tmp_dest = _dest.with_suffix(_dest.suffix + ".part")
        _run_cmd([
            "curl", "-L", "--retry", "3", "--retry-delay", "2",
            "--connect-timeout", "15", "-o", str(_tmp_dest), _url
        ])
        _tmp_dest.rename(_dest)
        _logs.append(f"✅ Downloaded: `{_dest.name}`")

    _nvcc_bin = shutil.which("nvcc")
    if not _nvcc_bin:
        raise RuntimeError("nvcc not found — CUDA toolkit missing")
    _CUDA_ROOT = Path(_nvcc_bin).resolve().parent.parent
    _CUDA_ENV = {
        "CUDACXX": _nvcc_bin,
        "CUDAToolkit_ROOT": str(_CUDA_ROOT),
    }

    def _ensure_build_tools():
        for _tool in ("cmake", "ninja"):
            if shutil.which(_tool):
                _logs.append(f"✅ Build tool available: {_tool}")
                continue
            _logs.append(f"⬇️ Installing build tool: {_tool} ...")
            _run_cmd([sys.executable, "-m", "pip", "install", "--quiet", _tool])

    def _ensure_cuda_libs():
        _libdir = _CUDA_ROOT / "lib"
        _created = 0
        for _name in sorted(os.listdir(_libdir)):
            if not _name.startswith("lib") or ".so." not in _name:
                continue
            _link = _libdir / (_name.split(".so.")[0] + ".so")
            if os.path.lexists(_link):
                continue
            os.symlink(_name, _link)
            _created += 1
        if _created:
            _logs.append(f"🔗 Created {_created} unversioned CUDA .so symlinks")

    def _ensure_cuda_cccl():
        _target = _CUDA_ROOT / "include" / "cccl"
        if _target.exists():
            _logs.append("✅ CUDA CCCL headers present")
            return
        _logs.append("⬇️ Installing CUDA CCCL headers ...")
        _run_cmd([sys.executable, "-m", "pip", "install", "--quiet", "nvidia-cuda-cccl"])
        import site as _site
        for _sp in _site.getsitepackages():
            _src = Path(_sp) / "nvidia" / "cu13" / "include" / "cccl"
            if _src.is_dir():
                os.symlink(_src, _target)
                _logs.append("🔗 Linked CUDA CCCL headers into toolkit")
                return
        raise RuntimeError("Could not locate CUDA CCCL headers after install")

    with mo.status.spinner(
        title="Preparing Patched llama-server & High-Speed HF Downloading..."
    ) as _spinner:
        _spinner.update("Step 0/3: Ensuring build tools & CUDA layout...")
        _ensure_build_tools()
        _ensure_cuda_libs()
        _ensure_cuda_cccl()

        _spinner.update("Step 1/3: Checking precompiled binary or compiling...")
        _precompiled_url = "https://github.com/ramin-mahmoodi/marimo-ai/releases/download/v1.0.0-blackwell/llama-server-blackwell-sm120.tar.gz"
        if not _llama_bin.exists():
            try:
                _test_req = requests.head(_precompiled_url, allow_redirects=True, timeout=5)
                if _test_req.status_code == 200:
                    _spinner.update("Downloading precompiled Blackwell llama-server from your GitHub Release...")
                    _tar_tmp = work_dir / "llama-server-blackwell-sm120.tar.gz"
                    _run_cmd(["curl", "-sSL", "-o", str(_tar_tmp), _precompiled_url])
                    _bin_dir = _llama_bin.parent
                    _bin_dir.mkdir(parents=True, exist_ok=True)
                    _run_cmd(["tar", "-xzf", str(_tar_tmp), "-C", str(_bin_dir)])
                    _llama_bin.chmod(0o755)
                    if _llama_bin.exists():
                        _logs.append("⚡ Loaded precompiled llama-server (Blackwell sm_120) from GitHub Release in 3 seconds!")
            except Exception:
                pass

        if not _llama_bin.exists():
            if not (_llama_dir / "CMakeLists.txt").exists():
                if _llama_dir.exists():
                    shutil.rmtree(_llama_dir)
                _llama_dir.mkdir(parents=True, exist_ok=True)

                _run_cmd(["git", "init"], _cwd=_llama_dir)
                _run_cmd(["git", "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git"], _cwd=_llama_dir)
                _run_cmd(["git", "fetch", "--depth", "1", "origin", _LLAMA_COMMIT], _cwd=_llama_dir)
                _run_cmd(["git", "checkout", "FETCH_HEAD"], _cwd=_llama_dir)

                _patch_path = _llama_dir / _PATCH_FILE
                _download_file(f"{_HF_BASE}/{_PATCH_FILE}", _patch_path, _min_bytes=500)
                _run_cmd(["git", "apply", "--check", str(_patch_path)], _cwd=_llama_dir)
                _run_cmd(["git", "apply", str(_patch_path)], _cwd=_llama_dir)
            else:
                _logs.append("♻️ Reusing existing llama.cpp source.")

            try:
                _run_cmd(
                    [
                        "cmake", "-G", "Ninja", "-S", ".", "-B", "build",
                        "-DGGML_CUDA=ON",
                        "-DCMAKE_CUDA_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DCMAKE_CUDA_ARCHITECTURES=native",
                        "-DLLAMA_BUILD_TESTS=OFF",
                        "-DLLAMA_BUILD_EXAMPLES=OFF",
                        "-DLLAMA_CURL=OFF",
                    ],
                    _cwd=_llama_dir,
                    _env=_CUDA_ENV,
                )
            except Exception:
                shutil.rmtree(_llama_dir / "build", ignore_errors=True)
                _run_cmd(
                    [
                        "cmake", "-G", "Ninja", "-S", ".", "-B", "build",
                        "-DGGML_CUDA=ON",
                        "-DCMAKE_CUDA_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DCMAKE_CUDA_ARCHITECTURES=89-virtual;90a-virtual",
                        "-DLLAMA_BUILD_TESTS=OFF",
                        "-DLLAMA_BUILD_EXAMPLES=OFF",
                        "-DLLAMA_CURL=OFF",
                    ],
                    _cwd=_llama_dir,
                    _env=_CUDA_ENV,
                )

            _run_cmd(
                ["cmake", "--build", "build", "--config", "Release", "--target", "llama-server", "-j", "4"],
                _cwd=_llama_dir,
            )
            _logs.append("✅ Compiled patched `llama-server` (HauhauCS-FastMTP enabled).")
        else:
            _logs.append("✅ Patched `llama-server` binary already built.")

        # Step 2: Download FastMTP sidecar (903 MB)
        _spinner.update("Step 2/3: High-speed download of FastMTP-32K sidecar (903 MB)...")
        _draft_path = _models_dir / _DRAFT_FILE
        _download_hf_model(_HF_REPO, _DRAFT_FILE, _draft_path, _min_bytes=800_000_000)

        # Step 2b: Vision Projector
        _mmproj_path = _models_dir / _MMPROJ_FILE
        if enable_vision_chk.value:
            _spinner.update("Step 2b/3: High-speed download of BF16 Vision Projector (931 MB)...")
            _download_hf_model(_HF_REPO, _MMPROJ_FILE, _mmproj_path, _min_bytes=800_000_000)

        # Step 3: Main Q8_K_P Model (31.46 GB) with HF-Transfer (Takes ~60-90 seconds)
        _spinner.update("Step 3/3: High-speed HF-Transfer download of Q8_K_P (31.46 GB)...")
        _model_path = _models_dir / _MODEL_FILE
        _download_hf_model(_HF_REPO, _MODEL_FILE, _model_path, _min_bytes=30_000_000_000)

    artifacts = {
        "llama_bin": str(_llama_bin),
        "model_path": str(_model_path),
        "draft_path": str(_draft_path),
        "mmproj_path": str(_mmproj_path) if enable_vision_chk.value else None,
        "ready": True,
    }

    _summary_md = "\n".join(f"- {_item}" for _item in _logs)
    mo.md(
        f"""
        ### 📦 Build & Model Assets Ready
        {_summary_md}

        ---
        {start_server_btn}
        """
    )
    return artifacts, start_server_btn


@app.cell
def _(mo):
    # 🌉 Public tunnel & Security Configuration
    # Paste your free zrok account token below, then click the Launch button.

    zrok_token_input = mo.ui.text(
        label="zrok account token",
        value="",
        kind="password",
        placeholder="paste the token from https://zrok.io",
        full_width=True,
    )

    zrok_name_input = mo.ui.text(
        label="Custom Static Domain Name (optional - keep empty for auto-generated ephemeral domain)",
        value="ramin-qwen",
        placeholder="e.g. ramin-qwen (lowercase, alphanumeric and hyphens)",
        full_width=True,
    )

    api_key_input = mo.ui.text(
        label="Custom API Key (Bearer Token Authorization)",
        value="sk-molab-local",
        placeholder="e.g. sk-my-secret-key (or keep default)",
        full_width=True,
    )

    mo.vstack(
        [
            mo.md(
                "### 🌉 Public tunnel via zrok & Security Settings\n"
                "1. Go to **https://zrok.io** → **Get Started** → create a free account (email only)\n"
                "2. Copy your account token and paste it into the field below\n"
                "3. *(Optional)* Set a custom static domain name (e.g. `ramin-qwen` → `https://ramin-qwen.share.zrok.io`)\n"
                "4. *(Optional)* Set your preferred API key (secures your endpoint from unauthorized web scans)\n"
                "5. Click button **2. Launch / Restart GPU Server & Public Tunnel**\n\n"
                "✅ When static name and API key are set, your endpoint and authorization stay identical across sessions!"
            ),
            zrok_token_input,
            zrok_name_input,
            api_key_input,
        ]
    )
    return (api_key_input, zrok_name_input, zrok_token_input)


@app.cell
def _(
    Path,
    api_key_input,
    artifacts,
    ctx_slider,
    depth_dropdown,
    enable_fastmtp_chk,
    mo,
    os,
    parallel_dropdown,
    re,
    reasoning_dropdown,
    requests,
    shutil,
    start_server_btn,
    state_store,
    subprocess,
    time,
    work_dir,
    zrok_name_input,
    zrok_token_input,
):
    mo.stop(
        not start_server_btn.value,
        mo.md(
            "ℹ️ *Click **2. Launch / Restart GPU Server & Public Tunnel** above to start serving.*"
        ),
    )

    def _stop_existing():
        for _key in ("cf_proc", "llama_proc"):
            _proc = state_store.get(_key)
            if _proc is not None:
                try:
                    _proc.terminate()
                    _proc.wait(timeout=5)
                except Exception:
                    try:
                        _proc.kill()
                    except Exception:
                        pass
                state_store[_key] = None
        subprocess.run(["pkill", "-f", "llama-server"], check=False)
        subprocess.run(["pkill", "-f", "zrok"], check=False)
        time.sleep(1)

    _stop_existing()

    _ZROK_VERSION = "2.0.9"
    _ZROK_TARBALL_URL = f"https://github.com/openziti/zrok/releases/download/v{_ZROK_VERSION}/zrok_{_ZROK_VERSION}_linux_amd64.tar.gz"

    def _ensure_zrok():
        """Return an absolute path to the zrok2 CLI, downloading it if necessary."""
        _existing = shutil.which("zrok2")
        if _existing:
            return _existing
        _bin_dir = Path("/usr/local/bin")
        if not os.access(_bin_dir, os.W_OK):
            _bin_dir = Path.home() / ".local" / "bin"
        _bin_dir.mkdir(parents=True, exist_ok=True)
        _target = _bin_dir / "zrok2"
        if _target.exists():
            return str(_target)

        import tarfile as _tarfile
        import tempfile as _tempfile

        _tmp_dir = Path(_tempfile.gettempdir()) / "zrok2_cli"
        _tmp_dir.mkdir(parents=True, exist_ok=True)
        _tarball = _tmp_dir / "zrok2.tar.gz"
        _dl = subprocess.run(
            ["curl", "-sSL", "-o", str(_tarball), _ZROK_TARBALL_URL],
            capture_output=True, text=True, timeout=600,
        )
        if _dl.returncode != 0:
            raise RuntimeError("Failed to download zrok2 CLI: " + (_dl.stderr or _dl.stdout))

        with _tarfile.open(_tarball, "r:gz") as _tf:
            _tf.extractall(_tmp_dir, filter="data")

        _src = _tmp_dir / "zrok2"
        if not _src.exists():
            raise RuntimeError("zrok2 binary missing inside downloaded archive")

        _src.chmod(0o755)
        try:
            shutil.copy2(_src, _target)
        except Exception:
            _target = Path.home() / ".local" / "bin" / "zrok2"
            _target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_src, _target)

        return str(_target)

    _llama_log_path = work_dir / "llama_server.log"

    _parallel_val = str(parallel_dropdown.value)
    _depth_val = str(depth_dropdown.value)
    _ctx_val = str(ctx_slider.value)
    _effort_val = str(reasoning_dropdown.value)

    # Standard continuous-batching flags + multi-slot parallelism + Hermes Agent compatibility
    _llama_cmd = [
        artifacts["llama_bin"],
        "--model",
        artifacts["model_path"],
        "--alias",
        "qwen3.8-27b",
        "--ctx-size",
        _ctx_val,
        "--parallel",
        _parallel_val,
        "--cont-batching",
        "--threads-http",
        "32",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--flash-attn",
        "on",
        "--no-mmap",
        "--temp",
        "1.0",
        "--top-k",
        "20",
        "--top-p",
        "0.95",
        "--min-p",
        "0",
        "--presence-penalty",
        "0",
        "--repeat-penalty",
        "1.0",
        "--jinja",
        "--reasoning",
        "on",
        "--reasoning-effort",
        _effort_val,
        "--reasoning-preserve",
        "--reasoning-format",
        "deepseek",
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
    ]

    # FastMTP Speculative Decoding: Enabled when user explicitly selects it (best for single-slot 1-turn stream)
    if enable_fastmtp_chk.value and artifacts.get("draft_path") and Path(artifacts["draft_path"]).exists():
        _llama_cmd.extend([
            "--spec-draft-model",
            artifacts["draft_path"],
            "--spec-draft-ngl",
            "all",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            _depth_val,
            "--spec-draft-p-min",
            "0",
        ])

    _api_key = (api_key_input.value or "sk-molab-local").strip()
    if _api_key:
        _llama_cmd.extend(["--api-key", _api_key])

    if artifacts.get("mmproj_path") and Path(artifacts["mmproj_path"]).exists():
        _llama_cmd.extend(["--mmproj", artifacts["mmproj_path"]])

    _env = os.environ.copy()
    _env["CUDA_VISIBLE_DEVICES"] = "0"

    with mo.status.spinner(
        title=f"Loading Q8_K_P ({_parallel_val} parallel slots) directly into 96 GB Blackwell VRAM (--no-mmap)..."
    ) as _sp:
        with open(_llama_log_path, "w", encoding="utf-8") as _llama_log_file:
            _llama_proc = subprocess.Popen(
                _llama_cmd,
                stdout=_llama_log_file,
                stderr=subprocess.STDOUT,
                env=_env,
            )
        state_store["llama_proc"] = _llama_proc

        # Wait until /health reports 200 OK
        _ready = False
        for _attempt in range(180):
            if _llama_proc.poll() is not None:
                _tail = _llama_log_path.read_text(encoding="utf-8", errors="ignore")[
                    -3000:
                ]
                raise RuntimeError(
                    f"llama-server exited unexpectedly (code {_llama_proc.returncode}):\n{_tail}"
                )
            try:
                _resp = requests.get("http://127.0.0.1:8081/health", timeout=2)
                if _resp.status_code == 200:
                    _ready = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not _ready:
            _tail = _llama_log_path.read_text(encoding="utf-8", errors="ignore")[-3000:]
            raise RuntimeError(
                f"Timed out waiting for llama-server health check:\n{_tail}"
            )

        _ztoken = (zrok_token_input.value or "").strip()
        _zrok_env_file = Path.home() / ".zrok2" / "environment.json"
        if not _ztoken and not _zrok_env_file.exists():
            raise RuntimeError("zrok token is empty — paste it in the tunnel setup cell above, then relaunch.")
        
        _zrok_bin = _ensure_zrok()
        if not _zrok_env_file.exists():
            _sp.update("Enabling zrok environment (first time)...")
            try:
                _en = subprocess.run(
                    [_zrok_bin, "enable", _ztoken, "--headless"],
                    capture_output=True, text=True, timeout=180,
                )
            except Exception as _enexc:
                raise RuntimeError("zrok enable timed out: " + str(_enexc)) from _enexc
            if _en.returncode != 0:
                raise RuntimeError(
                    "zrok enable failed (check your token):\n"
                    + (_en.stdout or "") + (_en.stderr or "")
                )
        
        _raw_name = (zrok_name_input.value or "").strip().lower()
        _zname = re.sub(r"[^a-z0-9-]", "", _raw_name)

        if _zname:
            _sp.update(f"Reserving static name '{_zname}' in zrok public namespace...")
            # 'zrok2 create name -n public <name>'
            # Ignore exit code in case the name was already created/reserved in a prior run
            subprocess.run(
                [_zrok_bin, "create", "name", "-n", "public", _zname],
                capture_output=True,
                text=True,
            )
            _share_args = [
                _zrok_bin,
                "share",
                "public",
                "http://127.0.0.1:8081",
                "-n",
                f"public:{_zname}",
                "--backend-mode",
                "proxy",
                "--headless",
                "--open",
            ]
        else:
            _share_args = [
                _zrok_bin,
                "share",
                "public",
                "http://127.0.0.1:8081",
                "--backend-mode",
                "proxy",
                "--headless",
                "--open",
            ]

        _sp.update("Starting zrok public tunnel...")
        _zrok_log_path = work_dir / "zrok_share.log"
        with open(_zrok_log_path, "w", encoding="utf-8") as _zrok_log_file:
            _cf_proc = subprocess.Popen(
                _share_args,
                stdout=_zrok_log_file,
                stderr=subprocess.STDOUT,
            )
        state_store["cf_proc"] = _cf_proc

        _public_url = None
        for _ in range(60):
            if _cf_proc.poll() is not None:
                break
            if _zrok_log_path.exists():
                _zt = _zrok_log_path.read_text(encoding="utf-8", errors="ignore")
                for _match in re.finditer(r"(?:https?://)?([a-zA-Z0-9._-]+\.shares?\.zrok\.io)", _zt):
                    _cand = _match.group(1).strip()
                    if not _cand.startswith("api."):
                        _public_url = f"https://{_cand}"
                        break
                if not _public_url and _zname and f"{_zname}.share" in _zt:
                    _public_url = f"https://{_zname}.shares.zrok.io"
                    break
                if _public_url:
                    break
            time.sleep(1)

        if not _public_url:
            _ztail = (
                _zrok_log_path.read_text(encoding="utf-8", errors="ignore")[-2500:]
                if _zrok_log_path.exists()
                else "no zrok log"
            )
            _err_detail = f" Could not bind static name '{_zname}' (it may be taken by another user)." if _zname else ""
            raise RuntimeError(f"Could not obtain zrok public URL.{_err_detail} zrok log:\n" + _ztail)

        state_store["public_url"] = _public_url

    endpoint_info = {
        "base_url": f"{_public_url}/v1",
        "chat_completions": f"{_public_url}/v1/chat/completions",
        "model": "qwen3.8-27b",
        "api_key": _api_key,
    }

    refresh_btn = mo.ui.run_button(label="🔄 Refresh VRAM & Server Logs", kind="neutral")

    mo.md(
        f"""
        ## 🟢 Server is ONLINE (100% in RTX Pro 6000 VRAM)

        ### 🔗 Connection Details for **Hermes**
        | Setting | Value |
        | :--- | :--- |
        | **OpenAI Base URL** | `{endpoint_info["base_url"]}` |
        | **Chat Completions URL** | `{endpoint_info["chat_completions"]}` |
        | **Model Name** | `{endpoint_info["model"]}` |
        | **API Key** | `{endpoint_info["api_key"]}` |

        ---
        ### ⚡ Quick Test Command (from your own machine)
        ```bash
        curl {endpoint_info["chat_completions"]} \\
          -H "Content-Type: application/json" \\
          -H "Authorization: Bearer {endpoint_info["api_key"]}" \\
          -d '{{
            "model": "{endpoint_info["model"]}",
            "messages": [{{"role": "user", "content": "سلام! وضعیت سرور چطوره؟"}}]
          }}'
        ```

        {refresh_btn}
        """
    )
    return endpoint_info, refresh_btn


@app.cell
def _(endpoint_info, mo, refresh_btn, requests, subprocess, work_dir):
    # Reference refresh_btn so clicking it re-evaluates this monitoring cell
    _ = refresh_btn.value

    _vram_status = "Unknown"
    try:
        _vram_status = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as _e:
        _vram_status = str(_e)

    _ram_status = "Unknown"
    try:
        _ram_status = subprocess.check_output(
            ["free", "-h"], text=True
        ).strip()
    except Exception:
        pass

    _health_status = "Offline"
    try:
        _r = requests.get("http://127.0.0.1:8081/health", timeout=2)
        _health_status = f"HTTP {_r.status_code}: {_r.text}"
    except Exception as _e:
        _health_status = f"Error: {_e}"

    _log_path = work_dir / "llama_server.log"
    _log_tail = ""
    if _log_path.exists():
        _lines = _log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        _log_tail = "\n".join(_lines[-25:])

    mo.md(
        f"""
        ### 📊 Live Hardware & Server Telemetry
        - **Server Health:** `{_health_status}`
        - **Active Endpoint:** `{endpoint_info["base_url"]}`
        - **GPU VRAM Usage:** `{_vram_status}`

        #### System RAM Usage (`free -h` - Protected by `--no-mmap`)
        ```text
        {_ram_status}
        ```

        #### Recent `llama-server` Logs (Last 25 lines)
        ```text
        {_log_tail}
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
