"""Run llama.cpp's server and talk to it.

Generated text is handed back to the caller. Nothing here prints or logs it.
"""

import socket
import subprocess
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

LLAMA_CPP_RELEASE = "b11524"


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

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def start(self, timeout: float = 600) -> None:
        if _port_in_use(self.port):
            raise ServerError(f"port {self.port} is already in use, so another server would answer instead")
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
        try:
            self._wait_until_ready(started, timeout)
            self._check_model()
            self._zero_default_scales()
        except BaseException:
            self.stop()
            raise

    def _wait_until_ready(self, started: float, timeout: float) -> None:
        assert self.process is not None
        while True:
            if self.process.poll() is not None:
                raise ServerError(f"llama-server exited with code {self.process.returncode}, see {self.log}")
            if self._healthy():
                return
            if time.perf_counter() - started > timeout:
                raise ServerError(f"llama-server wasn't ready after {timeout:.0f} s, see {self.log}")
            time.sleep(0.5)

    def _check_model(self) -> None:
        served = self.client.get("/props").json().get("model_path", "")
        if Path(served).resolve() != self.model.resolve():
            raise ServerError(f"the server on port {self.port} is serving {served}, not {self.model}")

    def _zero_default_scales(self) -> None:
        listed = self.client.get("/lora-adapters").json()
        self.adapter_ids = {Path(adapter["path"]).resolve(): adapter["id"] for adapter in listed}
        if listed:
            # --lora-init-without-apply leaves every adapter's default scale at 1, and a request that
            # names no adapters gets the defaults, so set them to 0 and check.
            zeros = [{"id": adapter["id"], "scale": 0.0} for adapter in listed]
            self.client.post("/lora-adapters", json=zeros).raise_for_status()
            if any(adapter["scale"] for adapter in self.client.get("/lora-adapters").json()):
                raise ServerError("couldn't set the adapters' default scale to 0")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def _healthy(self) -> bool:
        try:
            return self.client.get("/health").status_code == 200
        except httpx.TransportError:
            return False

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Cached prompt states don't record which adapters computed them, so reusing one can mix
        # an unedited prompt with an edited answer. cache_prompt false stops a slot reusing its own
        # cache, which the engine test checks. --cache-ram 0 also turns off the cache that slots
        # share; no test has shown it is needed on top, so it is kept only as a second guard.
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

    def chat_ids(self, chats: list[list[dict[str, str]]], template_kwargs: dict[str, Any]) -> list[list[int]]:
        """The token ids the chat endpoint feeds the model for each conversation, ready to answer.

        /apply-template renders a chat with the same code /v1/chat/completions uses, which then
        tokenizes the result with special tokens parsed and added; this does the same.
        """

        def ids(messages: list[dict[str, str]]) -> list[int]:
            body = {"messages": messages, "chat_template_kwargs": template_kwargs}
            rendered = self.client.post("/apply-template", json=body)
            rendered.raise_for_status()
            content = {"content": rendered.json()["prompt"], "add_special": True, "parse_special": True}
            tokenized = self.client.post("/tokenize", json=content)
            tokenized.raise_for_status()
            return tokenized.json()["tokens"]

        with ThreadPoolExecutor(self.parallel) as pool:
            return list(pool.map(ids, chats))

    def chat_prompt_length(self, messages: list[dict[str, str]], template_kwargs: dict[str, Any]) -> int:
        """How many prompt tokens a real chat request reports, to check chat_ids against."""
        body = {
            "messages": messages, "chat_template_kwargs": template_kwargs, "max_tokens": 1,
            "temperature": 0.0, "cache_prompt": False,
        }  # fmt: skip
        response = self.client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        return response.json()["usage"]["prompt_tokens"]

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
            {
                "prompt": p,
                "n_predict": 1,
                "n_probs": k,
                "temperature": 0.0,
                "lora": lora,
                "return_tokens": True,
            }
            for p in prompts
        ]
        return [
            self._first_token(payload, result)
            for payload, result in zip(payloads, self._complete_all(payloads), strict=True)
        ]

    def _first_token(self, payload: dict[str, Any], result: dict[str, Any]) -> dict[int, float]:
        # The server records no probabilities for a token that is an incomplete UTF-8 sequence.
        # Forbidding the tokens it picked makes it pick a complete one. The probabilities it then
        # reports come from the raw logits, before the bias, so they are still the model's own.
        banned: list[int] = []
        while not result.get("completion_probabilities"):
            if len(banned) == 20 or not result.get("tokens"):
                raise ServerError(
                    f"no first-token probabilities for a prompt of {len(payload['prompt'])} tokens"
                )
            banned.append(result["tokens"][0])
            result = self._complete({**payload, "logit_bias": [[token, False] for token in banned]})
        return {
            entry["id"]: entry["logprob"] for entry in result["completion_probabilities"][0]["top_logprobs"]
        }



def _port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0
