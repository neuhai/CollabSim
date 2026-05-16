#!/usr/bin/env python3
"""Launch an SGLang inference server for a HuggingFace model.

Usage examples:
    # Single GPU
    python scripts/start_sglang_server.py meta-llama/Llama-3.1-8B-Instruct

    # 4-GPU tensor parallelism on A100s
    python scripts/start_sglang_server.py meta-llama/Llama-3.3-70B-Instruct --tp 4

    # Custom port
    python scripts/start_sglang_server.py Qwen/Qwen2.5-7B-Instruct --port 30001

    # Multiple models on different ports (run in separate terminals):
    #   python scripts/start_sglang_server.py meta-llama/Llama-3.1-8B-Instruct --port 30000
    #   python scripts/start_sglang_server.py Qwen/Qwen2.5-7B-Instruct --port 30001
    #
    # Then set SGLANG_PORT=30000 (or 30001) before running the experiment.

The server exposes an OpenAI-compatible API at http://{host}:{port}/v1.
Set SGLANG_HOST and SGLANG_PORT in your environment (or .env file) to
point the experiment runner at the correct server.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error


def _wait_for_server(host: str, port: int, timeout_sec: int = 300) -> bool:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        attempt += 1
        wait = min(10, 2 * attempt)
        print(f"  waiting for server (attempt {attempt})...", flush=True)
        time.sleep(wait)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch an SGLang inference server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        help="HuggingFace model ID, e.g. meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--port", type=int, default=30000, help="Port to listen on.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        dest="tensor_parallel_size",
        help="Tensor parallel size (number of GPUs to use for this model).",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="Fraction of GPU memory reserved for static KV cache.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
        help="Model weight dtype.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Override maximum context length (tokens).",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["awq", "gptq", "fp8", None],
        help="Quantization scheme (omit for none).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for the server to become ready.",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--port", str(args.port),
        "--host", args.host,
        "--tp", str(args.tensor_parallel_size),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--dtype", args.dtype,
    ]
    if args.context_length is not None:
        cmd += ["--context-length", str(args.context_length)]
    if args.quantization is not None:
        cmd += ["--quantization", args.quantization]

    print("Starting SGLang server:")
    print(f"  {' '.join(cmd)}")
    print(f"  model:  {args.model}")
    print(f"  port:   {args.port}")
    print(f"  tp:     {args.tensor_parallel_size} GPU(s)")
    print(f"  dtype:  {args.dtype}")
    print()

    proc = subprocess.Popen(cmd)

    def _shutdown(sig: int, frame: object) -> None:
        print("\nShutting down SGLang server...", flush=True)
        proc.terminate()
        proc.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Waiting up to {args.startup_timeout}s for server to be ready...", flush=True)
    ready = _wait_for_server(args.host, args.port, timeout_sec=args.startup_timeout)
    if not ready:
        print(
            f"ERROR: Server did not become ready within {args.startup_timeout}s.",
            file=sys.stderr,
        )
        proc.terminate()
        sys.exit(1)

    print(f"\nServer ready at http://{args.host}:{args.port}/v1")
    print("Set these env vars before running an experiment:")
    print(f"  export SGLANG_HOST={args.host}")
    print(f"  export SGLANG_PORT={args.port}")
    print("\nPress Ctrl+C to stop.\n")

    proc.wait()


if __name__ == "__main__":
    main()
