"""Run llama.cpp's server and talk to it.

Generated text is handed back to the caller. Nothing here prints or logs it.
"""

import http.client
import json
import os
import re
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any

import httpx

LLAMA_CPP_RELEASE = "b11524"
# Linux refuses one command-line argument this long or longer (MAX_ARG_STRLEN counts the closing NUL).
MAX_ARGUMENT = 128 * 1024
# For requests that only read or change a setting; answers get _answering's longer limit.
QUICK = httpx.Timeout(30, connect=5)
# How llama.cpp logs a buffer the GPU had no room for, on CUDA and on Metal, and then that the buffer
# was the KV cache, or a hybrid model's recurrent state.
NO_ROOM = re.compile(
    r"allocating ([\d.]+) MiB on device \d+: cudaMalloc failed: out of memory"
    r"|error: failed to allocate buffer, size = +([\d.]+) MiB"
)
CACHE_FAILED = re.compile(r"failed to allocate buffer for (kv|rs) cache")
# How llama.cpp refuses a --device it can't see, as on a pod whose GPU has failed.
NO_DEVICE = re.compile(r"invalid device: (\S+)")


class ServerError(RuntimeError):
    pass


class Stopped(Exception):
    """A request ended because no one wants its answer any more."""


class Requests:
    """The requests behind one set of answers, each on a connection of its own, so stop() can end every
    one still running at once: llama-server drops a request whose connection closes, within a second,
    rather than finish a thinking no one will read."""

    def __init__(self) -> None:
        self.stopped = False
        self.open: set[http.client.HTTPConnection] = set()
        self.lock = Lock()

    def stop(self) -> None:
        with self.lock:
            self.stopped = True
            for connection in self.open:
                if connection.sock is not None:
                    with suppress(OSError):
                        connection.sock.shutdown(socket.SHUT_RDWR)


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
# The most tokens a model whose thinking budget ran out may take to finish a character it is part way
# through; llama-server waits as long as it takes, but a character takes at most 4 bytes.
UNFINISHED_CHARACTER = 8


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
        slots_setting: str | None = None,
    ):
        self.binary = binary.absolute()
        self.model = model.absolute()
        self.log = log
        self.adapters = [path.resolve() for path in adapters]
        # llama-server runs in the folder holding every adapter, so it can name each one briefly by its
        # path from there; or, where that isn't brief enough, in a temporary folder of links (see _link).
        self.workdir = Path(os.path.commonpath([p.parent for p in self.adapters])) if self.adapters else None
        self.names = [str(path.relative_to(self.workdir)) for path in self.adapters] if self.workdir else []
        self.links: tempfile.TemporaryDirectory | None = None
        self.parallel = parallel
        self.context = parallel * context_per_slot
        # What sets the number of slots, named when their KV cache doesn't fit.
        self.slots_setting = slots_setting
        self.port = port or _free_port()
        self.extra = list(extra)
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=QUICK)
        self.process: subprocess.Popen | None = None
        self.adapter_ids: dict[Path, int] = {}
        self._tokenized: dict[str, list[int]] = {}
        self._whole_tokens: dict[int, bool] = {}

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def start(self, timeout: float = 600) -> None:
        """Start llama-server and wait until it serves the model. Paths in extra must be absolute."""
        # Every request starts from position 0 (see _complete), so the RAM prompt cache and context
        # checkpoints are never used. Turning them off saves copying slot states to and from up to
        # 8 GiB of RAM, and on recurrent models such as Qwen3.5 an extra decode for every prompt.
        args = [
            str(self.binary), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
            "-np", str(self.parallel), "-c", str(self.context), "-ngl", "999", "--no-webui",
            "--cache-ram", "0", "--ctx-checkpoints", "0", *self.extra,
        ]  # fmt: skip
        try:
            if self.adapters:
                if len(self._lora_list().encode()) >= MAX_ARGUMENT:
                    self._link()
                args += ["--lora", self._lora_list(), "--lora-init-without-apply"]
            self.process = launch(args, self.log, timeout, self.workdir, self.slots_setting)
            self._check_model()
            self._zero_default_scales()
        except BaseException:
            self.stop()
            raise

    def _lora_list(self) -> str:
        """Every adapter in the one --lora argument llama.cpp wants, as a comma-separated list of their
        names in the server's folder, which keeps it short, as Linux limits the argument's length. Each
        is quoted, so a comma in a folder name can't split it."""
        return ",".join(map(_quoted, self.names))

    def _link(self) -> None:
        """Run the server in a temporary folder of links to the adapters, named by number, for adapters
        whose paths pass Linux's limit even from the folder holding them all, as a thinking model of 36
        layers' do once candor score adds its own. No adapter is renamed."""
        names = [f"{i}.gguf" for i in range(len(self.adapters))]
        size = len(",".join(map(_quoted, names)).encode())
        if size >= MAX_ARGUMENT:
            raise ServerError(
                f"the paths of {len(self.adapters)} adapters take {size} bytes, more than Linux allows in"
                f" one argument ({MAX_ARGUMENT - 1}), so load fewer adapters into one server"
            )
        self.links = tempfile.TemporaryDirectory(prefix="candor-adapters-")
        self.workdir, self.names = Path(self.links.name), names
        for name, path in zip(names, self.adapters, strict=True):
            (self.workdir / name).symlink_to(path)

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

    def stop(self) -> None:
        if self.process:
            terminate(self.process)
        self.client.close()
        if self.links is not None:
            self.links.cleanup()
            self.links = None

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

    def _complete(self, payload: dict[str, Any], requests: Requests | None = None) -> dict[str, Any]:
        # Cached prompt states don't record which adapters computed them, so reusing one can mix an
        # unedited prompt with an edited answer. cache_prompt false makes every request start from
        # position 0, which the engine test checks.
        body = {**payload, "cache_prompt": False}
        timeout = _answering(payload["n_predict"])
        if requests is None:
            return self._call("POST", "/completion", timeout, json=body)
        return self._stoppable(body, timeout.read or 0, requests)

    def _stoppable(self, body: dict[str, Any], timeout: float, requests: Requests) -> dict[str, Any]:
        """The server's answer to a /completion request that requests.stop() ends at once, on a connection
        of its own; Stopped once it has."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            with requests.lock:
                requests.open.add(connection)
            connection.connect()
            # Stopped while connecting, before stop() could see the connection.
            if requests.stopped:
                raise Stopped
            connection.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
        except (OSError, http.client.HTTPException) as error:
            if requests.stopped:
                raise Stopped from None
            raise self._failed(f"didn't answer /completion ({type(error).__name__})") from None
        finally:
            with requests.lock:
                requests.open.discard(connection)
            connection.close()
        if response.status >= 400:
            raise self._failed(f"answered /completion with HTTP {response.status}")
        return json.loads(data)

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
        requests = Requests()
        pool = ThreadPoolExecutor(self.parallel)
        try:
            futures = [
                pool.submit(self._answer_one, prompt, max_tokens, lora, sampling, seed, thinking, requests)
                for prompt, seed in zip(prompts, seeds, strict=True)
            ]
            for future in futures:
                yield future.result()
        except GeneratorExit:
            # The server drops a request whose connection closes, so close those still running, and
            # wait the moment that takes rather than let them slow whatever is asked next.
            requests.stop()
            pool.shutdown(cancel_futures=True)
            raise
        except BaseException:
            requests.stop()
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
        requests: Requests | None = None,
    ) -> Answer:
        """One answer. A thinking model thinks until it ends its thinking or its budget makes it stop,
        then answers in a request of its own, so the answer's length is limited apart from the
        thinking's. With requests, requests.stop() ends it, as no one wants the answer any more."""
        settings = {**sampling, "seed": seed, "lora": lora}
        if thinking is None:
            payload = {"prompt": prompt, "n_predict": max_tokens, **settings}
            result = self._checked(payload, self._complete(payload, requests))
            return Answer(result["content"], result["tokens_predicted"], result["stop_type"] != "limit")
        thought, closed, made_to_stop = self._think(prompt, settings, thinking, requests)
        if not closed:
            # The model ended without closing its thinking, so it never answered.
            return Answer("", 0, False, len(thought))
        payload = {"prompt": prompt + thought, "n_predict": max_tokens, **settings}
        result = self._checked(payload, self._complete(payload, requests))
        finished = result["stop_type"] != "limit"
        return Answer(result["content"], result["tokens_predicted"], finished, len(thought) - 1, made_to_stop)

    def thoughts(
        self,
        prompts: list[list[int]],
        sampling: Mapping[str, float],
        seeds: list[int],
        thinking: Thinking,
        done: Callable[[], None] | None = None,
    ) -> list[list[int] | None]:
        """Each prompt's thinking, up to and with its end tag, sampled with the prompt's seed; None where
        the model didn't end its thinking by itself: its budget ran out and it was made to stop, or it
        never closed it."""
        lora = self.lora(None)

        def one(prompt: list[int], seed: int) -> list[int] | None:
            thought, closed, made_to_stop = self._think(
                prompt, {**sampling, "seed": seed, "lora": lora}, thinking
            )
            return thought if closed and not made_to_stop else None

        return self._all(
            [partial(one, prompt, seed) for prompt, seed in zip(prompts, seeds, strict=True)], done
        )

    def _think(
        self,
        prompt: list[int],
        settings: dict[str, Any],
        thinking: Thinking,
        requests: Requests | None = None,
    ) -> tuple[list[int], bool, bool]:
        """A thinking model's thinking: the tokens it thought, up to and with its end tag when it closed
        its thinking; whether it closed it; and whether its budget ran out so it was made to stop.

        candor holds the model to its budget as llama-server's own budget does, rather than asking the
        server to, which would sample every token on the CPU: the server can't sample on the GPU with a
        budget. The model thinks freely to its budget; if it hasn't closed its thinking by then, it
        finishes a character it is part way through and is made to write the message and the close."""
        end = self.token_of(thinking.end)
        # What the server makes the model write when its budget runs out, built as the server does.
        forced = self.tokenize(thinking.message) + self.tokenize(thinking.close)
        if forced[-1] != end:
            raise ServerError(f"the thinking's close {thinking.close!r} doesn't end with its end tag")
        # The budget also counts the open's tokens after the start tag, as the server tokenizes it.
        freely = max(thinking.budget - (len(self.tokenize(thinking.opened)) - 1), 0)
        thought, stop = (
            self._thought(prompt, freely, settings, thinking, requests) if freely else ([], "limit")
        )
        if end in thought:
            return thought[: thought.index(end) + 1], True, False
        if stop != "limit":
            # The model ended without closing its thinking.
            return thought, False, False
        if thought and not self._whole(thought[-1]):
            # Finished in one request, so its tokens are drawn one after another as the server's
            # would be, and cut after the first whole one.
            more, stop = self._thought(prompt + thought, UNFINISHED_CHARACTER, settings, thinking, requests)
            # The server keeps the token that ended the model's turn, last.
            drawn = more[:-1] if stop == "eos" else more
            for i, token in enumerate(drawn):
                if token == end:
                    return thought + more[: i + 1], True, False
                if self._whole(token):
                    thought = thought + more[: i + 1]
                    break
            else:
                if stop != "limit":
                    return thought + more, False, False
                thought = thought + more
        return thought + forced, True, True

    def _thought(
        self,
        prompt: list[int],
        tokens: int,
        settings: dict[str, Any],
        thinking: Thinking,
        requests: Requests | None = None,
    ) -> tuple[list[int], str]:
        """Up to tokens more of the model's thinking, stopping at its end tag, and why it stopped:
        "limit" when it thought all of them."""
        payload = {
            "prompt": prompt,
            "n_predict": tokens,
            "stop": [thinking.end],
            "return_tokens": True,
            **settings,
        }
        result = self._checked(payload, self._complete(payload, requests))
        return result["tokens"], result["stop_type"]

    def _whole(self, token: int) -> bool:
        """Whether a token's text ends with a whole character, as llama-server's budget asks before it
        makes a model stop: the server shows a broken one as the replacement character."""
        if token not in self._whole_tokens:
            text = self._call("POST", "/detokenize", json={"tokens": [token]})["content"]
            self._whole_tokens[token] = not text.endswith("\ufffd")
        return self._whole_tokens[token]

    def token_of(self, tag: str) -> int:
        """The one token a think tag is in this model."""
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


def launch(
    args: list[str], log: Path, timeout: float = 600, cwd: Path | None = None, lower: str | None = None
) -> subprocess.Popen:
    """llama-server run with args, once it answers, its output added to log. If it exits first, the
    error says why (see _exited), naming lower, the setting that sizes its KV cache."""
    port = int(args[args.index("--port") + 1])
    if port_in_use(port):
        raise ServerError(f"port {port} is already in use, so another server would answer instead")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        logged = out.tell()
        process = subprocess.Popen(args, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)
    started = time.monotonic()
    try:
        while True:
            if process.poll() is not None:
                raise _exited(process, args, log, logged, lower)
            if _healthy(port):
                return process
            if time.monotonic() - started > timeout:
                raise ServerError(f"llama-server wasn't ready after {timeout:.0f} s, see {log}")
            # A small model is ready in a few hundredths of a second.
            time.sleep(0.05)
    except BaseException:
        terminate(process)
        raise


def _healthy(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200
    except httpx.TransportError:
        return False


def _exited(
    process: subprocess.Popen, args: list[str], log: Path, logged: int, lower: str | None
) -> ServerError:
    """Why the server exited as it started: that its GPU has failed or its KV cache or recurrent state
    didn't fit, when what it logged from byte logged on says so, or else its exit code. Nothing else of
    the log is shown."""
    with log.open("rb") as out:
        out.seek(logged)
        text = out.read().decode(errors="replace")
    if gone := NO_DEVICE.search(text):
        return ServerError(
            f"llama-server can't see {gone[1]}, the GPU it was started on, so the GPU has failed: pull "
            "the results, then replace the pod"
        )
    failed = CACHE_FAILED.search(text)
    # The buffer that failed is the last one logged before the cache's failure.
    found = NO_ROOM.findall(text, 0, failed.start()) if failed else []
    if failed and found:
        asked = float(found[-1][0] or found[-1][1])
        slots, context = (int(args[args.index(flag) + 1]) for flag in ("-np", "-c"))
        slots_of = f"{slots} slot" + ("s" if slots != 1 else "")
        cache = (
            f"KV cache for {slots_of} of {context // slots:,} tokens"
            if failed[1] == "kv"
            else f"recurrent state for {slots_of}"
        )
        # CUDA asks for the whole cache at once, Metal for it in parts.
        return ServerError(
            f"llama-server couldn't allocate its {cache}: it asked for a buffer of {asked:,.0f} MiB, "
            f"more GPU memory than it could get" + (f"; lower {lower}" if lower else "")
        )
    return ServerError(f"llama-server exited with code {process.returncode}, see {log}")


def terminate(process: subprocess.Popen, timeout: float = 60) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
