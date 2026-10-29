"""The pod watchdog and scripts/pod.sh, run against a fake pod: an ssh stand-in runs each command
here, with the pod's /root and /pod-* mapped into a folder, and Runpod's tools are faked."""

import base64
import contextlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from candor import config
from candor.run import Run

ROOT = Path(__file__).parent.parent
WATCHDOG = ROOT / "scripts" / "pod" / "watchdog.sh"
POD = ROOT / "scripts" / "pod.sh"
CURL = "curl -fsS -X DELETE -H Authorization: Bearer key https://api.runpod.io/v2/pods/pod1"
# A file time past the second, which rsync from macOS drops.
TIME = 1_700_000_000_123_456_789

SSH = r"""#!/bin/bash
while [[ $1 = -* ]]; do
    case $1 in -p | -o | -l | -i) shift 2 ;; *) shift ;; esac
done
shift
command="$*"
command=${command//\/root\//$FAKEPOD/root/}
command=${command//\/pod-/$FAKEPOD/pod-}
cd "$FAKEPOD/root" && HOME="$FAKEPOD/root" exec bash -c "$command"
"""
SETSID = f"#!{sys.executable}\nimport os, sys\nos.setsid()\nos.execvp(sys.argv[1], sys.argv[1:])\n"
# uv as its installer leaves it on the pod: it records how it was called, and a sync makes a venv
# whose python does the same.
UV = r"""#!/bin/bash
echo "uv $* in ${PWD#$FAKEPOD}" >>"$CALLS"
mkdir -p .venv/bin
printf '#!/bin/bash\necho "python $1 in ${PWD#$FAKEPOD}" >>"$CALLS"\n' >.venv/bin/python
chmod +x .venv/bin/python
"""


def tool(folder: Path, name: str, text: str) -> None:
    (folder / name).write_text(text)
    (folder / name).chmod(0o755)


def fake(folder: Path, name: str, works: bool) -> None:
    """A stand-in for a Runpod tool that records how it was called. Once a delete works the pod is
    gone, watchdog and all, so it stops the watchdog."""
    end = "kill $PPID\n" if works else "exit 1\n"
    tool(folder, name, f'#!/bin/bash\necho "{name} $*" >>"$CALLS"\n{end}')


def record(folder: Path, name: str) -> None:
    """A stand-in that records how it was called and does nothing else."""
    tool(folder, name, f'#!/bin/bash\necho "{name} $*" >>"$CALLS"\n')


def calls(folder: Path) -> list[str]:
    return (folder / "calls").read_text().splitlines() if (folder / "calls").exists() else []


def until(check, seconds: float = 10) -> None:
    for _ in range(int(seconds / 0.02)):
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("never happened")


def watch(
    tmp_path: Path,
    delete_at: int | None,
    deadline: str | None = None,
    runpodctl: bool = True,
    curl: bool = True,
    seconds: float = 1.0,
    count: int | None = None,
) -> tuple[list[str], str]:
    """Run the watchdog until it has made COUNT calls, or for SECONDS once it has written its deadline,
    checking every 0.05 s; return the calls it made and its deadline."""
    tools = tmp_path / "tools"
    tools.mkdir()
    fake(tools, "runpodctl", runpodctl)
    fake(tools, "curl", curl)
    file = tmp_path / "deadline"
    if deadline is not None:
        file.write_text(deadline)
    env = {
        "PATH": f"{tools}:/usr/bin:/bin",
        "CALLS": str(tmp_path / "calls"),
        "POD_DEADLINE_FILE": str(file),
        "POD_WATCHDOG_INTERVAL": "0.05",
        "RUNPOD_POD_ID": "pod1",
        "RUNPOD_API_KEY": "key",
    }
    if delete_at is not None:
        env["POD_DELETE_AT"] = str(delete_at)
    process = subprocess.Popen(["bash", str(WATCHDOG)], env=env, stdout=subprocess.DEVNULL)
    try:
        if count is None:
            until(lambda: file.exists() and file.read_text().strip() != "")
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=seconds)
        else:
            until(lambda: len(calls(tmp_path)) >= count)
    finally:
        process.kill()
        process.wait()
    return calls(tmp_path), file.read_text().strip()


def test_deletes_the_pod_once_its_deadline_passes(tmp_path):
    assert watch(tmp_path, int(time.time()) - 1, count=1)[0] == ["runpodctl pod delete pod1"]


def test_leaves_the_pod_before_its_deadline_and_records_it(tmp_path):
    later = int(time.time()) + 3600
    assert watch(tmp_path, later) == ([], str(later))


def test_a_moved_deadline_wins_either_way(tmp_path):
    now = int(time.time())
    (tmp_path / "later").mkdir()
    assert watch(tmp_path / "later", now - 1, deadline=str(now + 3600))[0] == []
    (tmp_path / "sooner").mkdir()
    sooner = watch(tmp_path / "sooner", now + 3600, deadline=str(now - 1), count=1)
    assert sooner[0] == ["runpodctl pod delete pod1"]


@pytest.mark.parametrize("deadline", [None, "12x", "0999999999", "99999999999999999999"])
def test_a_deadline_that_isnt_plain_seconds_deletes_the_pod(tmp_path, deadline):
    delete_at = int(time.time()) + 3600 if deadline else None
    assert watch(tmp_path, delete_at, deadline=deadline, count=1)[0] == ["runpodctl pod delete pod1"]


def test_falls_back_to_the_older_runpodctl_then_the_api(tmp_path):
    found, _ = watch(tmp_path, int(time.time()) - 1, runpodctl=False, count=3)
    assert found == ["runpodctl pod delete pod1", "runpodctl remove pod pod1", CURL]


def test_keeps_trying_until_a_delete_works(tmp_path):
    found, _ = watch(tmp_path, int(time.time()) - 1, runpodctl=False, curl=False, count=6)
    assert found[:6] == ["runpodctl pod delete pod1", "runpodctl remove pod pod1", CURL] * 2


class Pod:
    """A fake pod and the scripts/pod.sh that reaches it."""

    def __init__(self, tmp_path: Path, script: Path = POD):
        self.folder = tmp_path / "pod"
        (self.folder / "root").mkdir(parents=True)
        self.tools = tmp_path / "tools"
        self.tools.mkdir()
        tool(self.tools, "ssh", SSH)
        tool(self.tools, "setsid", SETSID)
        fake(self.tools, "runpodctl", True)
        fake(self.tools, "curl", True)
        self.script = script
        self.env = {
            "PATH": f"{self.tools}:/usr/bin:/bin:/usr/sbin",
            "HOME": str(tmp_path / "home"),
            "FAKEPOD": str(self.folder),
            "CANDOR_POD": "root@pod:22",
            "CALLS": str(self.folder / "calls"),
            "RUNPOD_POD_ID": "pod1",
            "POD_WATCHDOG_INTERVAL": "0.05",
            "POD_DEADLINE_FILE": str(self.folder / "pod-delete-at"),
        }
        self.container: subprocess.Popen[bytes] | None = None

    def run(self, *args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.script), *args], capture_output=True, text=True, env=self.env, cwd=cwd
        )

    def spawn(self, *args: str, cwd: Path, stderr: Path) -> subprocess.Popen[str]:
        with stderr.open("w") as out:
            return subprocess.Popen(
                [str(self.script), *args],
                stdout=subprocess.PIPE,
                stderr=out,
                text=True,
                env=self.env,
                cwd=cwd,
            )

    def boot(self, hours: str = "1", start: str | None = "exec sleep 600") -> None:
        """Start the container as Runpod would, with the command args gives and a fake /start.sh."""
        made = self.run("args", hours)
        assert made.returncode == 0, made.stderr
        [command] = json.loads(made.stdout)["cmd"]
        command = command.replace("/pod-", f"{self.folder}/pod-").replace(
            "/start.sh", f"{self.folder}/start.sh"
        )
        if start is not None:
            tool(self.folder, "start.sh", f"#!/bin/bash\n{start}\n")
        self.container = subprocess.Popen(
            ["bash", "-c", command], env=self.env, start_new_session=True, stderr=subprocess.DEVNULL
        )
        for _ in range(100):
            if (self.folder / "pod-delete-at").exists():
                return
            time.sleep(0.05)
        raise AssertionError("the watchdog never started")

    def close(self) -> None:
        # A group outlives its leader, and commands started on the pod lead groups of their own.
        groups = [int(pid.read_text()) for pid in (self.folder / "root" / "logs").glob("*.pid")]
        if self.container is not None:
            groups.append(self.container.pid)
        for group in groups:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
        subprocess.run(["pkill", "-f", str(self.folder)])


@pytest.fixture
def pod(tmp_path):
    made = Pod(tmp_path)
    yield made
    made.close()


def test_the_container_starts_runpods_services_and_this_watchdog(pod):
    pod.boot(start="echo started >$FAKEPOD/started; exec sleep 600")
    assert (pod.folder / "pod-watchdog.sh").read_bytes() == WATCHDOG.read_bytes()
    assert abs(int((pod.folder / "pod-delete-at").read_text()) - (time.time() + 3600)) < 10
    time.sleep(0.3)
    assert (pod.folder / "started").read_text() == "started\n"
    assert calls(pod.folder) == []


def test_the_container_outlives_a_failed_start_until_the_watchdog_deletes_it(pod):
    pod.boot(hours="0.001", start=None)
    time.sleep(1)
    assert pod.container is not None and pod.container.poll() is None
    pod.container.wait(timeout=10)
    assert calls(pod.folder) == ["runpodctl pod delete pod1"]


def test_args_print_how_the_command_is_built():
    made = subprocess.run([str(POD), "args", "2"], capture_output=True, text=True)
    container = json.loads(made.stdout)
    assert container["entrypoint"] == ["/bin/bash", "-c"]
    built = r"echo (\S+) \| base64 -d >/pod-watchdog\.sh && .* & /start\.sh; wait"
    encoded = re.fullmatch(built, container["cmd"][0])
    assert encoded and base64.b64decode(encoded[1]) == WATCHDOG.read_bytes()
    assert "the watchdog deletes the pod at" in made.stderr


@pytest.mark.parametrize("hours", ["0", "-1", "two", "", "49"])
def test_args_refuse_hours_out_of_range(hours):
    made = subprocess.run([str(POD), "args", hours], capture_output=True, text=True)
    assert made.returncode == 1 and made.stdout == ""
    assert "HOURS must be a number of hours above 0 and at most 48" in made.stderr


def test_commands_on_the_pod_need_its_address(pod):
    del pod.env["CANDOR_POD"]
    made = pod.run("ssh", "true")
    assert made.returncode == 1
    assert "set CANDOR_POD" in made.stderr


def test_deadline_moves_and_reports_it(pod):
    pod.boot()
    made = pod.run("deadline", "2")
    assert made.returncode == 0, made.stderr
    assert abs(int((pod.folder / "pod-delete-at").read_text()) - (time.time() + 7200)) < 10
    assert "the watchdog deletes the pod at" in made.stdout
    assert not (pod.folder / "pod-delete-at.new").exists()


def test_deadline_refuses_a_pod_without_a_watchdog(pod):
    made = pod.run("deadline")
    assert made.returncode == 1
    assert "the pod has no deadline" in made.stderr
    (pod.folder / "pod-delete-at").write_text(f"{int(time.time()) + 3600}\n")
    made = pod.run("deadline")
    assert made.returncode == 1
    assert "the watchdog isn't running" in made.stderr


def test_deadline_notices_the_watchdog_dying_while_the_container_runs(pod):
    pod.boot()
    assert pod.run("deadline").returncode == 0
    # Only the watchdog: the container's own shell names it too, and must keep running.
    subprocess.run(["pkill", "-fx", f"bash {pod.folder}/pod-watchdog.sh"], check=True)
    time.sleep(0.2)
    made = pod.run("deadline")
    assert made.returncode == 1
    assert "the watchdog isn't running" in made.stderr
    assert pod.container is not None and pod.container.poll() is None


def test_deadline_says_when_deletes_are_failing(pod):
    fake(pod.tools, "runpodctl", False)
    fake(pod.tools, "curl", False)
    pod.boot(hours="0.0003")
    time.sleep(1.5)
    made = pod.run("deadline")
    assert made.returncode == 1
    assert "the watchdog can't delete it" in made.stderr


def test_start_refuses_a_pod_without_a_watchdog(pod):
    made = pod.run("start", "nap", "sleep", "30")
    assert made.returncode == 1
    assert "the pod has no deadline" in made.stderr
    assert not (pod.folder / "root" / "logs").exists()


def test_start_log_and_stop_a_command(pod):
    pod.boot()
    (pod.folder / "root" / "candor").mkdir()
    made = pod.run("start", "nap", "bash", "-c", "echo 'it ran' with spaces; sleep 30")
    assert made.returncode == 0, made.stderr
    time.sleep(0.3)
    assert "it ran with spaces" in pod.run("log", "nap").stdout
    group = int((pod.folder / "root" / "logs" / "nap.pid").read_text())
    assert os.getpgid(group) == group
    again = pod.run("start", "nap", "sleep", "30")
    assert again.returncode == 1 and "nap is still running" in again.stderr
    assert int((pod.folder / "root" / "logs" / "nap.pid").read_text()) == group
    assert pod.run("stop", "nap").returncode == 0
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.killpg(group, 0)
    made = pod.run("stop", "nap")
    assert made.returncode == 1 and "nap isn't running" in made.stderr


@pytest.mark.parametrize("command", [["deadline"], ["start", "nap", "sleep", "1"], ["stop", "nap"]])
def test_a_pod_out_of_reach_is_reported_as_that(pod, command):
    tool(pod.tools, "ssh", "#!/bin/bash\nexit 255\n")
    made = pod.run(*command)
    assert made.returncode == 1
    assert "can't reach the pod over SSH" in made.stderr


def start(pod: Pod, name: str, *command: str) -> None:
    """Start COMMAND as NAME, and wait until it leads its own process group, as the real setsid has it
    do before start returns, or has ended."""
    made = pod.run("start", name, *command)
    assert made.returncode == 0, made.stderr
    pid = int((pod.folder / "root" / "logs" / f"{name}.pid").read_text())

    def formed() -> bool:
        try:
            return os.getpgid(pid) == pid
        except ProcessLookupError:
            return True

    until(formed)


def started(pod: Pod, *command: str) -> None:
    """Start COMMAND as nap on a pod with 12 hours left, with waits that check every 0.05 s."""
    pod.boot(hours="12")
    (pod.folder / "root" / "candor").mkdir()
    if not (pod.tools / "caffeinate").exists():
        tool(pod.tools, "caffeinate", "#!/bin/bash\n")
    pod.env["POD_WAIT_INTERVAL"] = "0.05"
    start(pod, "nap", *command)


def test_wait_keeps_the_mac_awake_until_the_command_ends(pod, tmp_path):
    tool(pod.tools, "caffeinate", '#!/bin/bash\necho "caffeinate $*" >>"$CALLS"\n')
    started(pod, "sleep", "1")
    waiting = pod.spawn("wait", "nap", cwd=tmp_path, stderr=tmp_path / "errors")
    out, _ = waiting.communicate(timeout=10)
    assert waiting.returncode == 0, (tmp_path / "errors").read_text()
    assert out == "nap exited with 0\n"
    until(lambda: any(line.startswith("caffeinate") for line in calls(pod.folder)))
    assert [line for line in calls(pod.folder) if line.startswith("caffeinate")] == [
        f"caffeinate -is -w {waiting.pid}"
    ]


def test_wait_rides_out_dropped_connections_but_not_a_pod_thats_gone(pod):
    pod.boot()
    (pod.folder / "root" / "candor").mkdir()
    tool(pod.tools, "caffeinate", "#!/bin/bash\n")
    pod.env["POD_WAIT_INTERVAL"] = "0.05"
    made = pod.run("wait", "nap")
    assert made.returncode == 1 and "nap was never started" in made.stderr
    start(pod, "nap", "sleep", "0")
    (pod.tools / "ssh").rename(pod.tools / "reach")
    tries = pod.folder / "tries"

    def drop(first: int) -> None:
        """The pod's SSH fails the first FIRST times it is asked, then works."""
        tries.write_text("")
        tool(
            pod.tools,
            "ssh",
            f"#!/bin/bash\necho >>{tries}\n(($(wc -l <{tries}) > {first})) || exit 255\n"
            f'exec {pod.tools}/reach "$@"\n',
        )

    drop(4)
    assert pod.run("wait", "nap").returncode == 0
    drop(5)
    made = pod.run("wait", "nap")
    assert made.returncode == 1 and "hasn't answered 5 times in a row" in made.stderr


@pytest.mark.parametrize(("command", "code"), [(["true"], 0), (["bash", "-c", "exit 3"], 3)])
def test_wait_exits_0_only_if_the_command_did(pod, command, code):
    started(pod, *command)
    made = pod.run("wait", "nap")
    assert made.returncode == (code != 0)
    assert f"nap exited with {code}" in made.stdout + made.stderr


def test_wait_says_when_the_command_was_stopped(pod):
    started(pod, "sleep", "30")
    assert pod.run("stop", "nap").returncode == 0
    made = pod.run("wait", "nap")
    assert made.returncode == 1 and "nap ended without an exit status" in made.stderr


def test_wait_pulls_once_more_at_the_end_and_says_how_the_command_ended_if_it_cant(pod, tmp_path):
    here = tmp_path / "here"
    here.mkdir()
    started(pod, "bash", "-c", "mkdir -p runs/r; echo done >runs/r/b")
    made = pod.run("wait", "nap", "runs/r", cwd=here)
    assert made.returncode == 0, made.stderr
    assert (here / "runs" / "r" / "b").read_text() == "done\n"
    start(pod, "failed", "bash", "-c", "exit 3")
    made = pod.run("wait", "failed", "runs/none", cwd=here)
    assert made.returncode == 1
    assert "failed exited with 3, but runs/none couldn't be pulled" in made.stderr


def test_wait_pulls_as_it_goes_and_rides_out_a_failed_pull(pod, tmp_path):
    here = tmp_path / "here"
    here.mkdir()
    code = pod.folder / "root" / "candor"
    started(pod, "bash", "-c", "until [ -f go ]; do sleep 0.05; done; echo late >runs/r/b")
    pod.env["POD_PULL_SECONDS"] = "0"
    errors = tmp_path / "errors"
    waiting = pod.spawn("wait", "nap", "runs/r", cwd=here, stderr=errors)
    try:
        # Nothing is there to pull yet, which doesn't end the wait.
        until(lambda: "couldn't pull runs/r this time" in errors.read_text())
        (code / "runs" / "r").mkdir(parents=True)
        (code / "runs" / "r" / "a").write_text("early")
        until(lambda: (here / "runs" / "r" / "a").exists())
        (code / "go").write_text("")
        out, _ = waiting.communicate(timeout=10)
    finally:
        waiting.kill()
        waiting.wait()
    assert waiting.returncode == 0, errors.read_text()
    assert out == "nap exited with 0\n"
    assert (here / "runs" / "r" / "b").read_text() == "late\n"


def test_wait_pulls_and_stops_as_the_deadline_nears(pod, tmp_path):
    here = tmp_path / "here"
    here.mkdir()
    started(pod, "sleep", "30")
    (pod.folder / "root" / "candor" / "runs" / "r").mkdir(parents=True)
    (pod.folder / "root" / "candor" / "runs" / "r" / "a").write_text("kept")
    pod.env["POD_WARN_SECONDS"] = str(13 * 3600)
    made = pod.run("wait", "nap", "runs/r", cwd=here)
    assert made.returncode == 1
    assert "the watchdog deletes the pod at" in made.stderr and "nap is still running" in made.stderr
    assert (here / "runs" / "r" / "a").read_text() == "kept"
    # Far enough away, the deadline doesn't end the wait.
    pod.env["POD_WARN_SECONDS"] = str(11 * 3600)
    start(pod, "quick", "true")
    made = pod.run("wait", "quick", cwd=here)
    assert made.returncode == 0, made.stderr
    # A deadline that isn't a time is one the watchdog is deleting the pod over.
    (pod.folder / "pod-delete-at").write_text("12x\n")
    made = pod.run("wait", "nap", cwd=here)
    assert made.returncode == 1 and "the pod's deadline isn't a time ('12x')" in made.stderr


def test_push_and_pull_files_and_folders(pod, tmp_path):
    here = tmp_path / "here"
    (here / "runs" / "r" / "gguf").mkdir(parents=True)
    (here / "runs" / "r" / "candor.toml").write_text("settings")
    (here / "runs" / "r" / "gguf" / "model.gguf").write_text("model")
    assert pod.run("push", "runs/r", cwd=here).returncode == 0
    code = pod.folder / "root" / "candor"
    assert (code / "runs" / "r" / "gguf" / "model.gguf").read_text() == "model"
    assert pod.run("push", "runs/r/candor.toml", "/root/elsewhere/c.toml", cwd=here).returncode == 0
    assert (pod.folder / "root" / "elsewhere" / "c.toml").read_text() == "settings"

    (code / "runs" / "r" / "exports" / "trial-1").mkdir(parents=True)
    (code / "runs" / "r" / "exports" / "trial-1" / "m.gguf").write_text("export")
    (code / "runs" / "r" / "measurements").mkdir()
    (code / "runs" / "r" / "measurements" / "base.json").write_text("{}")
    back = tmp_path / "back"
    assert pod.run("pull", "runs/r", str(back / "r"), cwd=here).returncode == 0
    assert (back / "r" / "measurements" / "base.json").read_text() == "{}"
    assert not (back / "r" / "gguf").exists() and not (back / "r" / "exports").exists()
    assert pod.run("pull", "runs/r/exports/trial-1", str(back / "t"), cwd=here).returncode == 0
    assert (back / "t" / "m.gguf").read_text() == "export"


def a_local_model(folder: Path) -> Path:
    """A model folder whose files have times past the second, as macOS gives them."""
    (folder / "weights").mkdir(parents=True)
    for index, name in enumerate(["config.json", "model.safetensors", "weights/extra.bin"]):
        (folder / name).write_text(name)
        os.utime(folder / name, ns=(TIME + index, TIME + index))
    return folder


def fingerprint(model: Path) -> str:
    return Run(model, config.RunConfig(model=config.Model(source=str(model)))).model_fingerprint


def test_a_local_model_pushed_keeps_its_fingerprint(pod, tmp_path):
    here = tmp_path / "here"
    model = a_local_model(here / "models" / "m")
    made = pod.run("push", "models/m", cwd=here)
    assert made.returncode == 0, made.stderr
    copy = pod.folder / "root" / "candor" / "models" / "m"
    assert fingerprint(copy) == fingerprint(model)
    assert (copy / "weights" / "extra.bin").stat().st_mtime_ns == TIME + 2
    # A file pushed to another name keeps its time too.
    assert pod.run("push", "models/m/config.json", "/root/elsewhere/c.json", cwd=here).returncode == 0
    assert (pod.folder / "root" / "elsewhere" / "c.json").stat().st_mtime_ns == TIME


def test_push_fails_where_the_pod_cant_keep_this_macs_times(pod, tmp_path):
    here = tmp_path / "here"
    a_local_model(here / "m")
    # The pod's disk keeps times only to the second.
    seconds = tmp_path / "seconds"
    seconds.mkdir()
    (seconds / "sitecustomize.py").write_text(
        "import os\n"
        "utime = os.utime\n"
        "os.utime = lambda path, ns: utime(path, ns=tuple(t // 10**9 * 10**9 for t in ns))\n"
    )
    (pod.tools / "ssh").rename(pod.tools / "reach")
    tool(pod.tools, "ssh", f'#!/bin/bash\nPYTHONPATH={seconds} exec {pod.tools}/reach "$@"\n')
    made = pod.run("push", "m", cwd=here)
    assert made.returncode == 1
    assert "3 of the pod's files don't have this Mac's size and time to the nanosecond" in made.stderr
    assert "a run would rebuild every stage" in made.stderr


def test_pull_stops_when_nothing_is_there_or_the_pod_cant_be_reached(pod, tmp_path):
    made = pod.run("pull", "runs/missing", str(tmp_path / "back"))
    assert made.returncode == 1 and "nothing at" in made.stderr
    # The connection drops only while pull asks whether the path is a folder.
    (pod.folder / "root" / "candor" / "runs" / "r" / "gguf").mkdir(parents=True)
    (pod.tools / "ssh").rename(pod.tools / "reach")
    tool(
        pod.tools,
        "ssh",
        f'#!/bin/bash\n[[ $* = *"echo folder"* ]] && exit 255\nexec {pod.tools}/reach "$@"\n',
    )
    made = pod.run("pull", "runs/r", str(tmp_path / "back"))
    assert made.returncode == 255
    assert not (tmp_path / "back").exists()


@pytest.mark.parametrize("path", ["a b", "runs/$HOME", "runs/*", "a;b"])
def test_paths_the_pods_shell_would_read_are_refused(pod, path):
    made = pod.run("push", path)
    assert made.returncode == 1 and "paths may hold only" in made.stderr


def git(repo: Path, *args: str) -> None:
    command = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    subprocess.run(command, check=True, capture_output=True)


def small_repo(tmp_path: Path) -> Path:
    """A repository of pod.sh, its watchdog, a.py and zzz.py, committed as 'start'."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "pod").mkdir(parents=True)
    shutil.copy(POD, repo / "scripts" / "pod.sh")
    shutil.copy(WATCHDOG, repo / "scripts" / "pod" / "watchdog.sh")
    (repo / ".gitignore").write_text("runs/\n.venv/\n*.gguf\n")
    for name in ("a.py", "zzz.py"):
        (repo / name).write_text(name)
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "start")
    return repo


def test_code_copies_the_working_tree_and_leaves_what_git_ignores(tmp_path):
    repo = small_repo(tmp_path)
    # The last file git lists is gone, which once stopped the copy without a word.
    (repo / "zzz.py").unlink()
    (repo / "new.py").write_text("new")
    (repo / "local.gguf").write_text("ignored")

    pod = Pod(tmp_path, script=repo / "scripts" / "pod.sh")
    (pod.folder / "root" / ".local" / "bin").mkdir(parents=True)
    tool(pod.folder / "root" / ".local" / "bin", "uv", UV)
    code = pod.folder / "root" / "candor"
    (code / "runs" / "r").mkdir(parents=True)
    (code / "runs" / "r" / "result.json").write_text("kept")
    (code / ".venv").mkdir()
    (code / "pushed.gguf").write_text("kept")
    (code / "stale.py").write_text("old")

    made = pod.run("code", cwd=repo)
    assert made.returncode == 0, made.stderr
    assert "start, with uncommitted changes" in made.stdout
    assert (code / "a.py").read_text() == "a.py" and (code / "new.py").read_text() == "new"
    assert not (code / "zzz.py").exists() and not (code / "stale.py").exists()
    assert not (code / "local.gguf").exists()
    assert (code / "runs" / "r" / "result.json").read_text() == "kept"
    assert (code / ".venv").is_dir() and (code / "pushed.gguf").read_text() == "kept"
    # A new uv.lock comes with the code, so the pod's packages follow it.
    assert calls(pod.folder) == ["uv sync --frozen --no-dev --quiet in /root/candor"]


def test_setup_arms_copies_installs_and_checks_the_gpu(tmp_path):
    repo = small_repo(tmp_path)
    pod = Pod(tmp_path, script=repo / "scripts" / "pod.sh")
    try:
        token = tmp_path / "home" / ".cache" / "huggingface" / "token"
        token.parent.mkdir(parents=True)
        token.write_text("hf_secret")
        # Nothing goes to a pod the watchdog won't delete.
        made = pod.run("setup", cwd=repo)
        assert made.returncode == 1 and "the pod has no deadline" in made.stderr
        assert sorted(path.name for path in (pod.folder / "root").iterdir()) == []

        pod.boot()
        tool(tmp_path, "uv", UV)
        installer = f"mkdir -p ~/.local/bin && cp {tmp_path / 'uv'} ~/.local/bin/uv"
        tool(pod.tools, "curl", f'#!/bin/bash\necho "curl $*" >>"$CALLS"\necho "{installer}"\n')
        record(pod.tools, "nvidia-smi")
        made = pod.run("setup", cwd=repo)
        assert made.returncode == 0, made.stderr
        copied = pod.folder / "root" / ".cache" / "huggingface" / "token"
        assert copied.read_text() == "hf_secret" and stat.S_IMODE(copied.stat().st_mode) == 0o600
        assert (pod.folder / "root" / "candor" / "a.py").read_text() == "a.py"
        installed, *checked = calls(pod.folder)
        assert installed.startswith("curl -LsSf https://astral.sh/uv/")
        assert checked == [
            "uv sync --frozen --no-dev --quiet in /root/candor",
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
            "python -c in /root/candor",
        ]
    finally:
        pod.close()


def test_code_refuses_while_a_command_is_running(tmp_path):
    repo = small_repo(tmp_path)
    pod = Pod(tmp_path, script=repo / "scripts" / "pod.sh")
    (pod.folder / "root" / ".local" / "bin").mkdir(parents=True)
    tool(pod.folder / "root" / ".local" / "bin", "uv", UV)
    (pod.folder / "root" / "logs").mkdir()
    job = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        (pod.folder / "root" / "logs" / "nap.pid").write_text(f"{job.pid}\n")
        made = pod.run("code", cwd=repo)
        assert made.returncode == 1 and "nap still running on the pod" in made.stderr
        assert not (pod.folder / "root" / "candor").exists()
    finally:
        job.kill()
        job.wait()
    made = pod.run("code", cwd=repo)
    assert made.returncode == 0, made.stderr
