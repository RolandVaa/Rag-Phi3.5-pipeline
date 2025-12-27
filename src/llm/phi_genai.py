from __future__ import annotations

import os
import time

import numpy as np
import onnxruntime_genai as og


def _add_windows_cuda_dll_dirs():
    """
    Windows-specific helper.

    On Windows, Python uses a restricted DLL search path. Even if CUDA is installed,
    dependent DLLs (e.g., cublas/cudnn) may not be found unless their folder is
    explicitly added to the DLL search directories.

    This adds: %CUDA_PATH%\\bin (if CUDA_PATH is set).
    Safe to call even if CUDA is not installed.
    """
    if os.name != "nt":
        return

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)


class PhiGenAI:
    """
    Lightweight wrapper for ONNX Runtime GenAI Phi models.

    Responsibilities:
      - Load Phi ONNX model with either CPU (default) or CUDA provider
      - Tokenize prompts and stream-decode tokens
      - Generate text while collecting performance metrics (TTFT, tok/s, etc.)
    """

    def __init__(self, model_dir: str, provider: str = "cpu", device_id: int = 0):
        # If the caller doesn't pass a model path, choose a sensible default
        # based on the selected accelerator.
        if not model_dir:
            if provider == "cuda":
                model_dir = r"models\Phi-3.5-mini-instruct-onnx\gpu\gpu-int4-awq-block-128"
            else:
                model_dir = r"models\Phi-3.5-mini-instruct-onnx\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4"

        self.model_dir = model_dir
        self.provider = provider
        self.device_id = device_id

        # Measure model initialization time (load + provider setup).
        t0 = time.perf_counter()

        if provider == "cuda":
            # Ensure CUDA DLLs are discoverable before ORT loads the CUDA provider.
            _add_windows_cuda_dll_dirs()

            # Explicitly configure ORT to use CUDA.
            cfg = og.Config(model_dir)
            cfg.clear_providers()
            cfg.append_provider("cuda")
            cfg.set_provider_option("cuda", "device_id", str(device_id))
            self.model = og.Model(cfg)
        else:
            # CPU path: allow GenAI to use default providers (CPU).
            self.model = og.Model(model_dir)

        # Tokenizer and stream decoder are tied to the loaded model.
        self.tok = og.Tokenizer(self.model)
        self.stream = self.tok.create_stream()

        # Expose load time and device type for benchmarking + sanity checks.
        self.load_s = time.perf_counter() - t0
        self.model_load_s = self.load_s
        self.device_type = getattr(self.model, "device_type", "unknown")

    def generate(
        self,
        prompt: str,
        max_new: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> dict:
        """
        Generate completion text and return key performance metrics.

        TTFT definition here ("normal TTFT"):
          - Start timing before prefill
          - Prefill occurs inside gen.append_tokens(...)
          - TTFT ends after the first generated token is produced

        Returned metrics:
          - prompt_tokens: number of input prompt tokens
          - new_tokens: number of generated tokens
          - ttft_s: time-to-first-token including prefill
          - gen_total_s: total generation time including prefill + decoding
          - tok_per_s: effective throughput (new_tokens / gen_total_s)
        """
        # Tokenize the prompt into int32 token IDs (required by GenAI).
        input_ids = np.asarray(self.tok.encode(prompt), dtype=np.int32)
        prompt_tokens = int(len(input_ids))

        # Configure generation/sampling settings.
        params = og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=prompt_tokens + max_new,  # total length = prompt + generated
            temperature=temperature,
            top_p=top_p,
        )

        # Measure "normal TTFT": prefill + first token.
        t_ttft0 = time.perf_counter()

        gen = og.Generator(self.model, params)

        # Prefill: processes the entire prompt to build KV-cache / internal state.
        # This is typically the expensive part for long RAG prompts (especially on CPU).
        gen.append_tokens(input_ids)

        # First generated token.
        gen.generate_next_token()
        token = gen.get_next_tokens()[0]
        first_piece = self.stream.decode(token)

        ttft_s = time.perf_counter() - t_ttft0

        # Continue decoding until completion.
        out = [first_piece]
        out_tokens = 1
        t_rest0 = time.perf_counter()

        while not gen.is_done():
            gen.generate_next_token()
            token = gen.get_next_tokens()[0]
            out_tokens += 1
            out.append(self.stream.decode(token))

        rest_s = time.perf_counter() - t_rest0

        # Total LLM time for this request (prefill + full decode).
        gen_total_s = ttft_s + rest_s
        tok_per_s = (out_tokens / gen_total_s) if gen_total_s > 0 else 0.0

        return {
            "text": "".join(out),
            "prompt_tokens": prompt_tokens,
            "new_tokens": int(out_tokens),
            "ttft_s": float(ttft_s),
            "gen_total_s": float(gen_total_s),
            "tok_per_s": float(tok_per_s),
        }
