"""The pod watchdog and scripts/pod.sh, run against a fake pod: an ssh stand-in runs each command
here, with the pod's /root and /pod-* mapped into a folder, and Runpod's tools are faked."""

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
WATCHDOG = ROOT / "scripts" / "pod" / "watchdog.sh"
POD = ROOT / "scripts" / "pod.sh"
CURL = "curl -fsS -X DELETE -H Authorization: Bearer key https://api.runpod.io/v2/pods/pod1"

SSH = r"""#!/bin/bash
while [[ $1 = -* ]]; do
    case $1 in -p | -o | -l | -i) shift 2 ;; *) shift ;; esac
done
shift
command="$*"
command=${command//\/root\//$FAKEPOD/root/}
command=${command//\/pod-/$FAKEPOD/pod-}
cd "$FAKEPOD/root" && exec bash -c "$command"
"""
SETSID = f"#!{sys.executable}\nimport os, sys\nos.setsid()\nos.execvp(sys.argv[1], sys.argv[1:])\n"


def tool(folder: Path, name: str, text: str) -> None:
    (folder / name).write_text(text)
    (folder / name).chmod(0o755)


def fake(folder: Path, name: str, works: bool) -> None:
    """A stand-in for a Runpod tool that records how it was called. Once a delete works the pod is
    gone, watchdog and all, so it stops the watchdog."""
    end = "kill $PPID\n" if works else "exit 1\n"
    tool(folder, name, f'#!/bin/bash\necho "{name} $*" >>"$CALLS"\n{end}')


def calls(folder: Path) -> list[str]:
    return (folder / "calls").read_text().splitlines() if (folder / "calls").exists() else []


def watch(
    tmp_path: Path,
    delete_at: int | None,
    deadline: str | None = None,
    runpodctl: bool = True,
    curl: bool = True,
    seconds: float = 1.0,
) -> tuple[list[str], str]:
    """Run the watchdog for a while, checking every 0.05 s; return the calls it made and its deadline."""
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
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    return calls(tmp_path), file.read_text().strip()


def test_deletes_the_pod_once_its_deadline_passes(tmp_path):
    assert watch(tmp_path, int(time.time()) - 1)[0] == ["runpodctl pod delete pod1"]


def test_leaves_the_pod_before_its_deadline_and_records_it(tmp_path):
    later = int(time.time()) + 3600
    assert watch(tmp_path, later) == ([], str(later))


def test_a_moved_deadline_wins_either_way(tmp_path):
    now = int(time.time())
    (tmp_path / "later").mkdir()
    assert watch(tmp_path / "later", now - 1, deadline=str(now + 3600))[0] == []
    (tmp_path / "sooner").mkdir()
    assert watch(tmp_path / "sooner", now + 3600, deadline=str(now - 1))[0] == ["runpodctl pod delete pod1"]


@pytest.mark.parametrize("deadline", [None, "12x", "0999999999", "99999999999999999999"])
def test_a_deadline_that_isnt_plain_seconds_deletes_the_pod(tmp_path, deadline):
    delete_at = int(time.time()) + 3600 if deadline else None
    assert watch(tmp_path, delete_at, deadline=deadline)[0] == ["runpodctl pod delete pod1"]


def test_falls_back_to_the_older_runpodctl_then_the_api(tmp_path):
    found, _ = watch(tmp_path, int(time.time()) - 1, runpodctl=False)
    assert found == ["runpodctl pod delete pod1", "runpodctl remove pod pod1", CURL]


def test_keeps_trying_until_a_delete_works(tmp_path):
    found, _ = watch(tmp_path, int(time.time()) - 1, runpodctl=False, curl=False, seconds=3)
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
        if self.container is not None and self.container.poll() is None:
            os.killpg(self.container.pid, signal.SIGKILL)
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
    subprocess.run(["pkill", "-f", f"bash {pod.folder}/pod-watchdog.sh"], check=True)
    time.sleep(0.2)
    made = pod.run("deadline")
    assert made.returncode == 1
    assert "the watchdog isn't running" in made.stderr


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


def test_code_copies_the_working_tree_and_leaves_what_git_ignores(tmp_path):
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
    # The last file git lists is gone, which once stopped the copy without a word.
    (repo / "zzz.py").unlink()
    (repo / "new.py").write_text("new")
    (repo / "local.gguf").write_text("ignored")

    pod = Pod(tmp_path, script=repo / "scripts" / "pod.sh")
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
