"""Run llama.cpp's server and talk to it.

Generated text is handed back to the caller. Nothing here prints or logs it.
"""

import os
import socket
import subprocess
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Event
from typing import Any

import httpx

LLAMA_CPP_RELEASE = "b11524"
# Linux refuses one command-line argument this long or longer (MAX_ARG_STRLEN counts the closing NUL).
MAX_ARGUMENT = 128 * 1024
# For requests that only read or change a setting; answers get _answering's longer limit.
QUICK = httpx.Timeout(30, connect=5)


class ServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Answer:
    """What the model answered, after any thinking."""

    text: str
    tokens: int
    # False when the answer reached its length limit before the model ended it, or a thinking model
    # ended without answering at all.
    finished: bool
    # A thinking model's thinking: how many tokens, and whether its budget ran out and it was made to
    # stop.
    thinking: int = 0
    forced: bool = False


@dataclass(frozen=True)
class Thinking:
    """How a thinking model thinks: every prompt ends by opening its thinking (opened, which starts
    with the start tag), and the end tag closes it. Once budget tokens are thought, llama-server makes
    the model write message and then close, the end tag with any text the template puts before it, as
    the server's chat endpoint does for the model's template; then it answers."""

    start: str
    end: str
    close: str
    opened: str
    budget: int
    message: str


# Answers without sampling settings are greedy: the likeliest token every time.
GREEDY = {"temperature": 0.0}


class LlamaServer:
    def __init__(
        self,
        binary: Path,
        model: Path,
        log: Path,
        adapters: Iterable[Path] = (),
        parallel: int = 16,
        context_per_slot: int = 1024,
        port: int | None = None,
        extra: Iterable[str] = (),
    ):
        self.binary = binary.absolute()
        self.model = model.absolute()
        self.log = log
        self.adapters = [path.resolve() for path in adapters]
        # llama-server runs in the folder holding every adapter, so it can name each one briefly.
        self.workdir = Path(os.path.commonpath([p.parent for p in self.adapters])) if self.adapters else None
        self.parallel = parallel
        self.context = parallel * context_per_slot
        self.port = port or _free_port()
        self.extra = list(extra)
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=QUICK)
        self.process: subprocess.Popen | None = None
        self.adapter_ids: dict[Path, int] = {}
        self._tokenized: dict[str, list[int]] = {}

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def start(self, timeout: float = 600) -> None:
        """Start llama-server and wait until it serves the model. Paths in extra must be absolute."""
        if port_in_use(self.port):
            raise ServerError(f"port {self.port} is already in use, so another server would answer instead")
        # Every request starts from position 0 (see _complete), so the RAM prompt cache and context
        # checkpoints are never used. Turning them off saves copying slot states to and from up to
        # 8 GiB of RAM, and on recurrent models such as Qwen3.5 an extra decode for every prompt.
        args = [
            str(self.binary), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
            "-np", str(self.parallel), "-c", str(self.context), "-ngl", "999", "--no-webui",
            "--cache-ram", "0", "--ctx-checkpoints", "0", *self.extra,
        ]  # fmt: skip
        if self.adapters:
            args += ["--lora", self._lora_list(), "--lora-init-without-apply"]
        started = time.perf_counter()
        with self.log.open("ab") as log:
            self.process = subprocess.Popen(args, cwd=self.workdir, stdout=log, stderr=subprocess.STDOUT)
        try:
            self._wait_until_ready(started, timeout)
            self._check_model()
            self._zero_default_scales()
        except BaseException:
            self.stop()
            raise

    def _lora_list(self) -> str:
        """Every adapter in the one --lora argument llama.cpp wants, as a comma-separated list.

        The paths are relative to the server's folder, because Linux limits the argument's length.
        Each is quoted, so a comma in a folder name can't split it.
        """
        assert self.workdir is not None
        value = ",".join(_quoted(str(path.relative_to(self.workdir))) for path in self.adapters)
        size = len(value.encode())
        if size >= MAX_ARGUMENT:
            raise ServerError(
                f"the paths of {len(self.adapters)} adapters take {size} bytes, more than Linux allows in"
                f" one argument ({MAX_ARGUMENT - 1}), so load fewer adapters into one server"
            )
        return value

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
        served = self._call("GET", "/props").get("model_path", "")
        if Path(served).resolve() != self.model.resolve():
            raise ServerError(f"the server on port {self.port} is serving {served}, not {self.model}")

    def _zero_default_scales(self) -> None:
        listed = self._call("GET", "/lora-adapters")
        workdir = self.workdir or Path.cwd()
        self.adapter_ids = {(workdir / adapter["path"]).resolve(): adapter["id"] for adapter in listed}
        if listed:
            # --lora-init-without-apply leaves every adapter's default scale at 1, and a request that
            # names no adapters gets the defaults, so set them to 0 and check.
            self._call("POST", "/lora-adapters", json=[{"id": a["id"], "scale": 0.0} for a in listed])
            if any(adapter["scale"] for adapter in self._call("GET", "/lora-adapters")):
                raise ServerError("couldn't set the adapters' default scale to 0")

    def stop(self, timeout: float = 60) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.client.close()

    def _healthy(self) -> bool:
        try:
            return self.client.get("/health", timeout=2).status_code == 200
        except httpx.TransportError:
            return False

    def _call(self, method: str, endpoint: str, timeout: httpx.Timeout = QUICK, **request: Any) -> Any:
        """The JSON the server answers with. A failure names the endpoint and points at the log."""
        try:
            response = self.client.request(method, endpoint, timeout=timeout, **request)
        except httpx.TransportError as error:
            raise self._failed(f"didn't answer {endpoint} ({type(error).__name__})") from None
        if response.is_error:
            raise self._failed(f"answered {endpoint} with HTTP {response.status_code}")
        return response.json()

    def _failed(self, what: str) -> ServerError:
        code = self.process.poll() if self.process else None
        state = "is still running" if code is None else f"has exited with code {code}"
        return ServerError(f"llama-server {what} and {state}, see {self.log}")

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Cached prompt states don't record which adapters computed them, so reusing one can mix an
        # unedited prompt with an edited answer. cache_prompt false makes every request start from
        # position 0, which the engine test checks.
        body = {**payload, "cache_prompt": False}
        return self._call("POST", "/completion", _answering(payload["n_predict"]), json=body)

    def _complete_all(
        self, payloads: list[dict[str, Any]], done: Callable[[], None] | None = None
    ) -> list[dict[str, Any]]:
        """Every payload's result, in order. done is called as each one finishes."""
        return self._all([partial(self._complete, payload) for payload in payloads], done)

    def _all[T](self, work: Sequence[Callable[[], T]], done: Callable[[], None] | None = None) -> list[T]:
        """Every piece of work's result, in order, as many at once as the server has slots. done is
        called as each one finishes."""
        results: list[Any] = [None] * len(work)
        pool = ThreadPoolExecutor(self.parallel)
        try:
            futures = {pool.submit(job): i for i, job in enumerate(work)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                if done:
                    done()
        except BaseException:
            # A failed request or a signal stops the rest at once, rather than sending every one still
            # queued before the server can be stopped.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
        return results

    def lora(self, scales: Mapping[Path, float] | None) -> list[dict[str, float]]:
        """The request's adapter list. Adapters left out run at scale 0, which llama.cpp skips."""
        return [
            {"id": self.adapter_ids[path.resolve()], "scale": scale}
            for path, scale in (scales or {}).items()
            if scale
        ]

    def tokenize(self, text: str) -> list[int]:
        """Text as the model's token ids, special tokens such as a think tag included, kept for reuse."""
        if text not in self._tokenized:
            body = {"content": text, "add_special": False, "parse_special": True}
            self._tokenized[text] = self._call("POST", "/tokenize", json=body)["tokens"]
        return self._tokenized[text]

    def apply_template(self, messages: list[dict[str, str]], template_kwargs: dict[str, Any]) -> str:
        """The prompt the chat endpoint renders for a conversation, ready for the answer."""
        body = {"messages": messages, "chat_template_kwargs": template_kwargs}
        return self._call("POST", "/apply-template", json=body)["prompt"]

    def chat_ids(
        self, chats: list[list[dict[str, str]]], template_kwargs: dict[str, Any], then: str = ""
    ) -> list[list[int]]:
        """The token ids the chat endpoint feeds the model for each conversation, ready to answer, with
        then written after each, such as the text that closes a thinking model's thinking.

        /apply-template renders a chat with the same code /v1/chat/completions uses, which then
        tokenizes the result with special tokens parsed and added; this does the same.
        """

        def ids(messages: list[dict[str, str]]) -> list[int]:
            prompt = self.apply_template(messages, template_kwargs) + then
            body = {"content": prompt, "add_special": True, "parse_special": True}
            return self._call("POST", "/tokenize", json=body)["tokens"]

        with ThreadPoolExecutor(self.parallel) as pool:
            return list(pool.map(ids, chats))

    def chat_prompt_length(self, messages: list[dict[str, str]], template_kwargs: dict[str, Any]) -> int:
        """How many prompt tokens a real chat request reports, to check chat_ids against."""
        body = {
            "messages": messages, "chat_template_kwargs": template_kwargs, "max_tokens": 1,
            "temperature": 0.0, "cache_prompt": False,
        }  # fmt: skip
        return self._call("POST", "/v1/chat/completions", _answering(1), json=body)["usage"]["prompt_tokens"]

    def generate(
        self,
        prompts: list[list[int]],
        max_tokens: int,
        scales: Mapping[Path, float] | None = None,
        done: Callable[[], None] | None = None,
        sampling: Mapping[str, float] = GREEDY,
        seeds: list[int] | None = None,
        thinking: Thinking | None = None,
    ) -> list[Answer]:
        """Answers to prompts given as token ids, max_tokens long at most after any thinking, sampled
        with each prompt's seed. done is called as each one finishes."""
        lora = self.lora(scales)
        work = [
            partial(self._answer_one, prompt, max_tokens, lora, sampling, seed, thinking)
            for prompt, seed in zip(prompts, seeds or [0] * len(prompts), strict=True)
        ]
        return self._all(work, done)

    def answers(
        self,
        prompts: list[list[int]],
        max_tokens: int,
        scales: Mapping[Path, float] | None = None,
        sampling: Mapping[str, float] = GREEDY,
        seeds: list[int] | None = None,
        thinking: Thinking | None = None,
    ) -> Generator[Answer]:
        """generate's answers in order, each as soon as it is ready. The server answers the rest
        meanwhile. Closing the iterator early cancels those not yet started."""
        lora = self.lora(scales)
        seeds = seeds or [0] * len(prompts)
        stopped = Event()
        pool = ThreadPoolExecutor(self.parallel)
        try:
            futures = [
                pool.submit(self._answer_one, prompt, max_tokens, lora, sampling, seed, thinking, stopped)
                for prompt, seed in zip(prompts, seeds, strict=True)
            ]
            for future in futures:
                yield future.result()
        except GeneratorExit:
            # The server can't be told to drop an answer it has started, so wait for those rather
            # than let them slow whatever is asked next; a thinking model's that are still thinking
            # don't go on to answer.
            stopped.set()
            pool.shutdown(cancel_futures=True)
            raise
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()

    def _answer_one(
        self,
        prompt: list[int],
        max_tokens: int,
        lora: list[dict[str, float]],
        sampling: Mapping[str, float],
        seed: int,
        thinking: Thinking | None,
        stopped: Event | None = None,
    ) -> Answer:
        """One answer. A thinking model thinks until it ends its thinking or its budget makes it stop,
        then answers in a request of its own, so the answer's length is limited apart from the
        thinking's; unless stopped is set by then, as no one wants the answer any more."""
        settings = {**sampling, "seed": seed, "lora": lora}
        if thinking is None:
            payload = {"prompt": prompt, "n_predict": max_tokens, **settings}
            result = self._checked(payload, self._complete(payload))
            return Answer(result["content"], result["tokens_predicted"], result["stop_type"] != "limit")
        end = self._single(thinking.end)
        # What the server makes the model write when its budget runs out, built as the server does.
        forced = self.tokenize(thinking.message) + self.tokenize(thinking.close)
        if forced[-1] != end:
            raise ServerError(f"the thinking's close {thinking.close!r} doesn't end with its end tag")
        payload = {
            "prompt": prompt,
            # The sampler may wait a few tokens to finish a character before it makes the model stop.
            "n_predict": thinking.budget + len(forced) + 8,
            "reasoning_budget_tokens": thinking.budget,
            "reasoning_budget_start_tag": thinking.start,
            "reasoning_budget_end_tags": list(dict.fromkeys([thinking.close, thinking.end])),
            "reasoning_budget_message": thinking.message,
            "generation_prompt": thinking.opened,
            "preserved_tokens": [thinking.start, thinking.end],
            "stop": [thinking.end],
            "return_tokens": True,
            **settings,
        }
        thought = self._checked(payload, self._complete(payload))["tokens"]
        if end not in thought:
            if len(thought) >= payload["n_predict"]:
                raise ServerError(
                    f"llama-server didn't hold a thinking model to its budget of {thinking.budget}"
                )
            # The model ended without closing its thinking, so it never answered.
            return Answer("", 0, False, len(thought))
        closed = thought.index(end) + 1
        # The budget also counts the open's tokens after the start tag, as the server tokenizes it.
        opened = len(self.tokenize(thinking.opened)) - 1
        thought_freely = max(thinking.budget - opened, 0)
        made_to_stop = (
            closed >= thought_freely + len(forced) and thought[closed - len(forced) : closed] == forced
        )
        if stopped is not None and stopped.is_set():
            return Answer("", 0, False, closed - 1, made_to_stop)
        payload = {"prompt": prompt + thought[:closed], "n_predict": max_tokens, **settings}
        result = self._checked(payload, self._complete(payload))
        finished = result["stop_type"] != "limit"
        # The thinking's length leaves out the end tag.
        return Answer(result["content"], result["tokens_predicted"], finished, closed - 1, made_to_stop)

    def _single(self, tag: str) -> int:
        ids = self.tokenize(tag)
        if len(ids) != 1:
            raise ServerError(f"the think tag {tag!r} isn't one token in this model but {len(ids)}")
        return ids[0]

    def _checked(self, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        # The server also flags an answer that fills its slot exactly, so only a short one counts.
        if result["truncated"] and result["tokens_predicted"] < payload["n_predict"]:
            length = len(payload["prompt"])
            raise ServerError(
                f"the server's context cut an answer short: a prompt of {length} tokens needs"
                f" a slot of at least {length + payload['n_predict']} tokens"
            )
        return result

    def top_logprobs(
        self,
        prompts: list[list[int]],
        k: int,
        scales: Mapping[Path, float] | None = None,
        done: Callable[[], None] | None = None,
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
            for payload, result in zip(payloads, self._complete_all(payloads, done), strict=True)
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


def _quoted(field: str) -> str:
    """A field of a llama.cpp comma-separated list, kept whole whatever commas or quotes it holds."""
    return '"' + field.replace('"', '""') + '"'


def _answering(tokens: int) -> httpx.Timeout:
    """A limit for a request that answers only once its whole answer is ready: 10 minutes plus a
    second per token, so only a server that has stopped working can reach it."""
    return httpx.Timeout(600 + tokens, connect=5)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0
