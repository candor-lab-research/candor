"""Run llama.cpp's server and talk to it.

Generated text is handed back to the caller. Nothing here prints or logs it.
"""

import subprocess
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx


class ServerError(RuntimeError):
    pass


class LlamaServer:
    def __init__(
        self,
        binary: Path,
        model: Path,
        log: Path,
        adapters: Iterable[Path] = (),
        parallel: int = 16,
        context_per_slot: int = 1024,
        port: int = 8080,
        extra: Iterable[str] = (),
    ):
        self.binary = binary
        self.model = model
        self.log = log
        self.adapters = [path.resolve() for path in adapters]
        self.parallel = parallel
        self.context = parallel * context_per_slot
        self.port = port
        self.extra = list(extra)
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=None)
        self.process: subprocess.Popen | None = None
        self.adapter_ids: dict[Path, int] = {}
        self.startup_seconds = 0.0

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def start(self, timeout: float = 600) -> None:
        args = [
            str(self.binary), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
            "-np", str(self.parallel), "-c", str(self.context), "-ngl", "999", "--no-webui",
            "--cache-ram", "0", *self.extra,
        ]  # fmt: skip
        if self.adapters:
            args += ["--lora", ",".join(map(str, self.adapters)), "--lora-init-without-apply"]
        started = time.perf_counter()
        with self.log.open("ab") as log:
            self.process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        while not self._healthy():
            if self.process.poll() is not None:
                raise ServerError(f"llama-server exited with code {self.process.returncode}, see {self.log}")
            if time.perf_counter() - started > timeout:
                self.stop()
                raise ServerError(f"llama-server wasn't ready after {timeout:.0f} s, see {self.log}")
            time.sleep(0.5)
        self.startup_seconds = time.perf_counter() - started
        listed = self.client.get("/lora-adapters").json()
        self.adapter_ids = {Path(adapter["path"]).resolve(): adapter["id"] for adapter in listed}
        if listed:
            # --lora-init-without-apply leaves every adapter's default scale at 1, and a request that
            # names no adapters gets the defaults, so set them to 0 and check.
            zeros = [{"id": adapter["id"], "scale": 0.0} for adapter in listed]
            self.client.post("/lora-adapters", json=zeros).raise_for_status()
            if any(adapter["scale"] for adapter in self.client.get("/lora-adapters").json()):
                self.stop()
                raise ServerError("couldn't set the adapters' default scale to 0")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=60)

    def _healthy(self) -> bool:
        try:
            return self.client.get("/health").status_code == 200
        except httpx.TransportError:
            return False

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Cached prompt states don't record which adapters computed them, so reusing one can mix
        # an unedited prompt with an edited answer. --cache-ram 0 covers the cache shared by slots.
        response = self.client.post("/completion", json={**payload, "cache_prompt": False})
        if response.status_code != 200:
            raise ServerError(f"/completion returned HTTP {response.status_code}, see {self.log}")
        return response.json()

    def _complete_all(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(self.parallel) as pool:
            return list(pool.map(self._complete, payloads))

    def lora(self, scales: Mapping[Path, float] | None) -> list[dict[str, float]]:
        """The request's adapter list. Adapters left out run at scale 0, which llama.cpp skips."""
        return [
            {"id": self.adapter_ids[path.resolve()], "scale": scale}
            for path, scale in (scales or {}).items()
            if scale
        ]

    def generate(
        self, prompts: list[list[int]], max_tokens: int, scales: Mapping[Path, float] | None = None
    ) -> list[str]:
        """Greedy answers to prompts given as token ids."""
        lora = self.lora(scales)
        payloads = [{"prompt": p, "n_predict": max_tokens, "temperature": 0.0, "lora": lora} for p in prompts]
        return [result["content"] for result in self._complete_all(payloads)]

    def top_logprobs(
        self, prompts: list[list[int]], k: int, scales: Mapping[Path, float] | None = None
    ) -> list[dict[int, float]]:
        """The k most likely first answer tokens for each prompt, as {token id: log probability}."""
        lora = self.lora(scales)
        payloads = [
            {"prompt": p, "n_predict": 1, "n_probs": k, "temperature": 0.0, "lora": lora} for p in prompts
        ]
        return [
            {entry["id"]: entry["logprob"] for entry in result["completion_probabilities"][0]["top_logprobs"]}
            for result in self._complete_all(payloads)
        ]

    def throughput(
        self, prompts: list[list[int]], tokens: int, scales: Mapping[Path, float] | None = None
    ) -> float:
        """Generated tokens per second with every answer forced to full length. The text is discarded."""
        lora = self.lora(scales)
        payloads = [
            {"prompt": p, "n_predict": tokens, "ignore_eos": True, "temperature": 0.0, "lora": lora}
            for p in prompts
        ]
        started = time.perf_counter()
        results = self._complete_all(payloads)
        return sum(result["tokens_predicted"] for result in results) / (time.perf_counter() - started)
