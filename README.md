# Phi-3.5 RAG CLI Demo (Windows) — CPU vs CUDA

This repository contains a command-line (CLI) Retrieval-Augmented Generation (RAG) demo that runs **Phi-3.5-mini-instruct** locally/on-device using **ONNX Runtime GenAI**. The app ingests documents (PDF/TXT/MD), builds a vector index, retrieves relevant chunks for a user question, and generates an answer grounded in the retrieved context.

Two inference accelerators are supported:
- **CPU** (default)
- **NVIDIA CUDA GPU** (enabled with `--provider cuda`)

---

## Repository Structure

```
rag-phi35/
  src/
    cli.py
    llm/
      phi_genai.py
    rag/
      chunking.py
      ingest.py
      query.py
      store.py
  kb/                # place PDFs/TXT/MD here
  models/            # download models here (not committed to GitHub)
  data/index/        # created after ingest (not committed)
  requirements_cpu.txt
  requirements_cuda.txt
```

---

## Prerequisites (Machine Setup)

### Required
- **Windows 11 x64** (Windows 10 should also work)
- **Python 3.11.x (64-bit)** installed and available on PATH

### For CUDA / GPU runs
- NVIDIA GPU driver installed
- **CUDA Toolkit 12.x** (tested with **CUDA 12.5**)
- **NVIDIA cuDNN** (required DLLs must be discoverable by the CUDA provider)

### Only needed if recreating environments (fallback)
- **Microsoft C++ Build Tools (MSVC)**  
  Needed only if you must compile/install `hnswlib` from source on Windows.

---

## PowerShell Execution Policy (if activation is blocked)

If activation scripts are blocked, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Path Length Note (Windows)

To avoid Windows path-length issues, clone/extract into a short path such as:

```text
C:\work\rag-phi35\
```

---

## Get the Model: `microsoft/Phi-3.5-mini-instruct-onnx`

This demo expects the ONNX INT4 AWQ variants from the Hugging Face repository:

- **Repo:** `microsoft/Phi-3.5-mini-instruct-onnx`

The demo looks for these folders:

- CPU model:
  - `models\Phi-3.5-mini-instruct-onnx\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4`
- GPU model:
  - `models\Phi-3.5-mini-instruct-onnx\gpu\gpu-int4-awq-block-128`

### Option A (recommended): Download via `huggingface_hub` (no Git LFS)

1) Install the downloader:
```powershell
pip install -U huggingface_hub
```

2) Download the repo snapshot into `models/`:
```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/Phi-3.5-mini-instruct-onnx', local_dir='models/Phi-3.5-mini-instruct-onnx', local_dir_use_symlinks=False)"
```

> If you are prompted for authentication due to model gating, log in:
```powershell
huggingface-cli login
```

### Option B: Download via Git + Git LFS

```powershell
git lfs install
git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx models/Phi-3.5-mini-instruct-onnx
```

---

## Python Environments

You can either:
- Use the ZIP-submission environments (`.venv_cpu`, `.venv_cuda`) if included, **or**
- Recreate them using `requirements_cpu.txt` and `requirements_cuda.txt`.

### Create/Recreate CPU environment
```powershell
python -m venv .venv_cpu
.\.venv_cpu\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements_cpu.txt
```

### Create/Recreate CUDA environment
```powershell
python -m venv .venv_cuda
.\.venv_cuda\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements_cuda.txt
```

---

## Running the Demo

All commands should be run from the repository root folder (the folder containing `src/`).

### 1) Ingest documents (build the vector index)

Put your documents into `kb/` (PDF/TXT/MD). Example:
```text
kb\nasa_systems_engineering_handbook.pdf
```

Activate the CPU environment and run ingest:
```powershell
cd <PATH_TO_REPO_ROOT>
.\.venv_cpu\Scripts\Activate.ps1
python -m src.cli ingest --docs-dir kb --out-dir data/index
```

### 2) Ask a question on CPU (default)

```powershell
.\.venv_cpu\Scripts\Activate.ps1
python -m src.cli ask "What is the difference between product verification and product validation?"
```

### 3) Ask a question on CUDA GPU

```powershell
.\.venv_cuda\Scripts\Activate.ps1
python -m src.cli ask "What is the difference between product verification and product validation?" --provider cuda
```

---

## CLI Output Metrics

For each query, the CLI prints:
- `device_type` (CPU or CUDA) and `model_load_s`
- `prompt_tokens`, `new_tokens`
- `ttft_s`, `gen_total_s`, `tok/s`

---

## Troubleshooting

### CUDA provider fails to load (missing DLLs)

If you see missing DLL errors (e.g., `cudnn64_9.dll` or `cublasLt64_12.dll`), verify:
- CUDA Toolkit 12.x is installed
- cuDNN is installed and its DLLs are discoverable at runtime
- `CUDA_PATH` points to the correct CUDA install (e.g., `...\CUDA\v12.5`)
- `"%CUDA_PATH%\bin"` contains the required CUDA DLLs

Quick checks:
```powershell
echo $env:CUDA_PATH
dir "$env:CUDA_PATH\bin\cublasLt64_12.dll"
```

### `hnswlib` install fails (only if rebuilding env)

If `pip install hnswlib` fails with a Visual C++ error, install **Microsoft C++ Build Tools** and retry.

---

## Notes

- This demo uses ONNX Runtime GenAI with an ONNX-exported Phi-3.5 model for efficient local inference.
- RAG retrieval uses sentence-transformer embeddings and an HNSW approximate nearest-neighbor index for fast top-k retrieval.
