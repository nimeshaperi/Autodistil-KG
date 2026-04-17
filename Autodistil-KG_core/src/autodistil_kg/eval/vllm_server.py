"""vLLM server manager — start/stop a local vLLM OpenAI-compatible server.

Launches vLLM as a subprocess with the base model and optional LoRA adapter,
waits for it to be healthy, then exposes the server URL for predictors.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8432
HEALTH_TIMEOUT = 600  # 10 min — first run may download model weights


class VLLMServerManager:
    """Manages a local vLLM server subprocess.

    Starts vLLM with the given base model and optional LoRA adapter,
    waits for the health endpoint to respond, and stops it on cleanup.
    """

    def __init__(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
        port: int = DEFAULT_PORT,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.model_name = model_name
        self.lora_path = lora_path
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self._process: Optional[subprocess.Popen] = None
        self._log_path: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def lora_model_name(self) -> str:
        """The model name to use for LoRA-adapted inference."""
        return "finetuned"

    def start(self) -> str:
        """Start the vLLM server and wait for it to become healthy.

        Returns the base URL (e.g. ``http://localhost:8432/v1``).
        """
        if self._process is not None:
            logger.warning("vLLM server already running (pid=%d)", self._process.pid)
            return self.base_url

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--trust-remote-code",
            "--dtype", "auto",
        ]

        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        if self.lora_path:
            cmd.extend([
                "--enable-lora",
                "--lora-modules", f"{self.lora_model_name}={self.lora_path}",
            ])

        logger.info("Starting vLLM server: %s", " ".join(cmd))

        # Write vLLM output to a log file so we can read it for errors
        self._log_path = f"/tmp/vllm_server_{self.port}.log"
        log_file = open(self._log_path, "w")

        self._process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for health check
        health_url = f"http://localhost:{self.port}/health"
        start_time = time.time()
        last_log_line = 0
        while time.time() - start_time < HEALTH_TIMEOUT:
            # Check if process crashed
            if self._process.poll() is not None:
                exit_code = self._process.returncode
                log_file.close()
                output = self._read_log_tail(2000)
                self._process = None
                raise RuntimeError(
                    f"vLLM server exited with code {exit_code}.\n"
                    f"Last output:\n{output}"
                )

            # Log progress from vLLM output
            try:
                with open(self._log_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[last_log_line:]:
                        stripped = line.strip()
                        if stripped:
                            logger.info("vLLM: %s", stripped[:200])
                    last_log_line = len(lines)
            except Exception:
                pass

            # Health check
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    elapsed = int(time.time() - start_time)
                    logger.info(
                        "vLLM server healthy at %s (pid=%d, started in %ds)",
                        self.base_url, self._process.pid, elapsed,
                    )
                    self._warmup()
                    return self.base_url
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(5)

        # Timeout — collect logs for error message
        output = self._read_log_tail(2000)
        self.stop()
        raise RuntimeError(
            f"vLLM server did not become healthy within {HEALTH_TIMEOUT}s.\n"
            f"Last output:\n{output}"
        )

    def _warmup(self) -> None:
        """Send a short request to trigger CUDA graph compilation.

        The first real inference on vLLM compiles CUDA graphs and is
        significantly slower.  A throwaway warmup request absorbs that
        cost so actual evaluation predictions have consistent latency.
        """
        model_name = self.lora_model_name if self.lora_path else self.model_name
        try:
            logger.info("Warming up vLLM with a short request...")
            t0 = time.time()
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
                timeout=120,
            )
            resp.raise_for_status()
            elapsed = round(time.time() - t0, 1)
            logger.info("vLLM warmup complete (%.1fs)", elapsed)
        except Exception as e:
            logger.warning("vLLM warmup request failed (non-fatal): %s", e)

    def _read_log_tail(self, max_chars: int = 2000) -> str:
        """Read the last N chars from the vLLM log file."""
        if not self._log_path or not Path(self._log_path).exists():
            return "(no log file)"
        try:
            with open(self._log_path, "r") as f:
                content = f.read()
            return content[-max_chars:] if len(content) > max_chars else content
        except Exception:
            return "(could not read log)"

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._process is None:
            return

        logger.info("Stopping vLLM server (pid=%d)", self._process.pid)
        try:
            self._process.terminate()
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not terminate gracefully, killing")
            self._process.kill()
            self._process.wait(timeout=5)
        except Exception as e:
            logger.warning("Error stopping vLLM server: %s", e)
        finally:
            self._process = None

    def __del__(self) -> None:
        self.stop()
