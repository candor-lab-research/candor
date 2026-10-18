#!/bin/bash
# Runs candor on a Runpod pod from this Mac. The pod is created and deleted with Runpod's own tools;
# this arms its watchdog, sets it up and moves files to and from it. Run it from the repository root.
#
#   scripts/pod.sh args HOURS        the container command to create the pod with: Runpod's own
#                                    start, and a watchdog that deletes the pod HOURS from now
#   scripts/pod.sh setup             check the watchdog is armed, copy the code and the Hugging Face
#                                    token, install candor, and check torch and llama-server see the GPU
#   scripts/pod.sh code              copy this working tree's code again, uncommitted changes too,
#                                    and bring the pod's packages in line with it
#   scripts/pod.sh push SRC [DEST]   copy a file or folder to the pod (DEST defaults to SRC)
#   scripts/pod.sh pull PATH [DEST]  copy a file or folder back (DEST defaults to PATH), leaving
#                                    out gguf/ and exports/ folders unless PATH is one of them
#   scripts/pod.sh start NAME CMD..  run a command in the code folder, detached, logged as NAME
#   scripts/pod.sh log NAME          the end of that log
#   scripts/pod.sh stop NAME         stop that command
#   scripts/pod.sh wait NAME [PATH]  wait for that command to end, keeping this Mac awake until it
#                                    does, or until the pod stops answering or nears its deadline;
#                                    pulling PATH every 30 minutes and at the end; exits 0 only if
#                                    the command did
#   scripts/pod.sh deadline [HOURS]  check the watchdog and when it deletes the pod, or move that to
#                                    HOURS from now
#   scripts/pod.sh ssh [CMD..]       a shell or a command on the pod
#
# Every command but args needs CANDOR_POD, the pod's direct SSH address: root@IP:PORT. Relative
# paths on the pod are under /root/candor, the code folder, as they are under this repository here.
set -euo pipefail

UV_VERSION=0.11.28
MAX_HOURS=48
CODE=/root/candor
LOGS=/root/logs
DEADLINE=/pod-delete-at
KNOWN_HOSTS=${CANDOR_POD_KNOWN_HOSTS:-$HOME/.cache/candor/pod-known-hosts}
TOKEN=$HOME/.cache/huggingface/token
here=$(cd "$(dirname "$0")/.." && pwd)

die() {
    echo "pod.sh: $*" >&2
    exit 1
}

need() {
    (($1)) || die "usage: scripts/pod.sh $2"
}

seconds() {
    awk -v h="$1" -v most="$MAX_HOURS" \
        'BEGIN { if (h !~ /^[0-9]*\.?[0-9]+$/ || h <= 0 || h > most) exit 1; printf "%d", h * 3600 }' ||
        die "HOURS must be a number of hours above 0 and at most $MAX_HOURS, not '$1'"
}

when() {
    date -r "$1" '+%Y-%m-%d %H:%M %Z'
}

plain() {
    # Paths go to the pod's shell, so only characters it takes literally.
    [[ $1 =~ ^[A-Za-z0-9._/+-]+$ ]] || die "paths may hold only letters, digits and . _ / + -, not '$1'"
}

named() {
    [[ $1 =~ ^[A-Za-z0-9._-]+$ ]] || die "NAME takes letters, digits, '.', '_' and '-', not '$1'"
}

connect() {
    [[ ${CANDOR_POD:-} =~ ^[^@]+@[^:]+:[0-9]+$ ]] || die "set CANDOR_POD to the pod's SSH address, root@IP:PORT"
    host=${CANDOR_POD%:*}
    port=${CANDOR_POD##*:}
    mkdir -p "$(dirname "$KNOWN_HOSTS")"
    ssh_command="ssh -p $port -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS -o ServerAliveInterval=30 -o ConnectTimeout=30"
}

on_pod() {
    $ssh_command "$host" "$@"
}

reach() {
    # The command prints what it finds, so only a failure to reach the pod fails it.
    on_pod "$1" || die "can't reach the pod over SSH ($CANDOR_POD): check it is up and try again"
}

remote() {
    plain "$1"
    if [[ $1 = /* ]]; then echo "$1"; else echo "$CODE/$1"; fi
}

args() {
    local wait at watchdog
    wait=$(seconds "$1") || exit 1
    at=$(($(date +%s) + wait))
    watchdog=$(base64 <"$here/scripts/pod/watchdog.sh" | tr -d '\n')
    # If /start.sh fails or ends, the shell waits on the watchdog, so the container stays up until
    # the watchdog deletes the pod, rather than restarting without one.
    printf '{"entrypoint": ["/bin/bash", "-c"], "cmd": ["%s"]}\n' \
        "echo $watchdog | base64 -d >/pod-watchdog.sh && POD_DELETE_AT=$at setsid bash /pod-watchdog.sh >>/pod-watchdog.log 2>&1 </dev/null & /start.sh; wait"
    echo "the watchdog deletes the pod at $(when "$at")" >&2
}

armed() {
    local at alive
    at=$(reach "cat $DEADLINE 2>/dev/null || echo none")
    [[ $at != none ]] ||
        die "the pod has no deadline: it wasn't created with scripts/pod.sh args, so delete it yourself"
    # The whole command line, as the container's own shell and this check's both mention the file.
    alive=$(reach "pgrep -fx 'bash /pod-watchdog.sh' >/dev/null && echo yes || echo no")
    [[ $alive = yes ]] || die "the watchdog isn't running: delete the pod yourself"
    [[ $at =~ ^[1-9][0-9]{0,10}$ ]] || die "the pod's deadline isn't a time ('$at'), so the watchdog is deleting it"
    (($(date +%s) < at)) ||
        die "the deadline passed at $(when "$at") but the pod is still up, so the watchdog can't delete it" \
            "(see /pod-watchdog.log): delete it yourself"
    echo "the watchdog deletes the pod at $(when "$at")"
}

deadline() {
    local wait at
    if [[ $# -gt 0 ]]; then
        wait=$(seconds "$1") || exit 1
        at=$(($(date +%s) + wait))
        # Written whole and moved into place, so the watchdog never reads a half-written file.
        on_pod "echo $at >$DEADLINE.new && mv $DEADLINE.new $DEADLINE"
    fi
    armed
}

running() {
    # The commands started on the pod that are still running, by name.
    reach "for pid in $LOGS/*.pid; do
        [ -f \$pid ] && kill -0 -- -\$(cat \$pid) 2>/dev/null && basename \$pid .pid
    done; true"
}

code() {
    local busy
    # Packages and code changed under a running command can break its next import.
    busy=$(running)
    [[ -z $busy ]] || die "$(echo $busy) still running on the pod: stop it or wait for it first"
    # Every file git tracks or would, as it is here. A tracked file deleted here isn't copied.
    stage=$(mktemp -d)
    trap 'rm -rf "$stage"' EXIT
    git -C "$here" ls-files -z --cached --others --exclude-standard |
        while IFS= read -r -d '' file; do
            if [[ -e $here/$file ]]; then printf '%s\0' "$file"; fi
        done |
        tar --null -T - -c -C "$here" | tar -x -C "$stage"
    # Whatever git ignores, such as runs/ and .venv/, is left as it is on the pod.
    rsync -a --delete --exclude-from "$here/.gitignore" -e "$ssh_command" "$stage/" "$host:$CODE/"
    on_pod "cd $CODE && ~/.local/bin/uv sync --frozen --no-dev --quiet"
    local changes=""
    [[ -z $(git -C "$here" status --porcelain) ]] || changes=", with uncommitted changes"
    echo "code: $(git -C "$here" log -1 --format='%h %s')$changes"
}

setup() {
    [[ -f $TOKEN ]] || die "no Hugging Face token at $TOKEN: run 'hf auth login'"
    # A new pod can have an old one's address.
    ssh-keygen -R "[${host#*@}]:$port" -f "$KNOWN_HOSTS" >/dev/null 2>&1 || true
    armed
    on_pod "command -v rsync >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null)"
    on_pod "[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh >/dev/null"
    code
    on_pod "umask 077 && mkdir -p ~/.cache/huggingface && cat >~/.cache/huggingface/token" <"$TOKEN"
    on_pod "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
    on_pod "cd $CODE && .venv/bin/python -c '
import torch
from candor import toolchain
assert torch.cuda.is_available(), \"torch sees no GPU\"
print(\"torch\", torch.__version__, \"sees\", torch.cuda.get_device_name())
print(\"llama-server sees the GPU:\", toolchain.binaries())
'"
}

push() {
    local dest
    dest=$(remote "${2:-$1}")
    plain "$1"
    [[ -e $1 ]] || die "nothing at $1"
    if [[ -d $1 ]]; then
        on_pod "mkdir -p $(printf '%q' "$dest")"
        rsync -a --partial -e "$ssh_command" "${1%/}/" "$host:$dest/"
    else
        on_pod "mkdir -p $(printf '%q' "$(dirname "$dest")")"
        rsync -a --partial -e "$ssh_command" "$1" "$host:$dest"
    fi
}

pull() {
    local source kind dest=${2:-$1}
    source=$(remote "${1%/}")
    plain "$dest"
    local quoted
    quoted=$(printf '%q' "$source")
    kind=$(on_pod "if [ -d $quoted ]; then echo folder; elif [ -e $quoted ]; then echo file; fi")
    case $kind in
    folder)
        mkdir -p "$dest"
        rsync -a --partial --exclude gguf/ --exclude exports/ -e "$ssh_command" "$host:$source/" "${dest%/}/"
        ;;
    file)
        mkdir -p "$(dirname "$dest")"
        rsync -a --partial -e "$ssh_command" "$host:$source" "$dest"
        ;;
    *) die "nothing at $source on the pod" ;;
    esac
}

start() {
    local name=$1
    shift
    named "$name"
    armed
    # setsid makes the shell running the command lead a process group, which stop ends as a whole. The
    # shell writes the command's exit status, which a command stopped never gets to.
    on_pod "set -e; mkdir -p $LOGS; cd $CODE
        if [ -f $LOGS/$name.pid ] && kill -0 -- -\$(cat $LOGS/$name.pid) 2>/dev/null; then
            echo '$name is still running: stop it first' >&2
            exit 1
        fi
        rm -f $LOGS/$name.exit
        PATH=$CODE/.venv/bin:\$PATH setsid nohup sh -c '\"\$@\"; echo \$? >$LOGS/$name.exit' sh \
            $(printf '%q ' "$@") >$LOGS/$name.log 2>&1 </dev/null &
        echo \$! >$LOGS/$name.pid"
    echo "started; scripts/pod.sh log $name"
}

stop() {
    local result
    named "$1"
    result=$(reach "kill -TERM -- -\$(cat $LOGS/$1.pid 2>/dev/null) 2>/dev/null && echo stopped || echo absent")
    [[ $result = stopped ]] || die "$1 isn't running"
    echo "stopped $1"
}

deleting() {
    # Why the watchdog deletes the pod soon, from its deadline, or nothing.
    if ! [[ $1 =~ ^[1-9][0-9]{0,10}$ ]]; then
        echo "the pod's deadline isn't a time ('$1'), so the watchdog is deleting it"
    elif (($1 - $(date +%s) < ${POD_WARN_SECONDS:-21600})); then
        echo "the watchdog deletes the pod at $(when "$1")"
    fi
}

wait_for() {
    local name=$1 path=${2:-} misses=0 reply state at code result soon pulled=$SECONDS
    named "$name"
    [[ -z $path ]] || plain "$path"
    # Ends by itself when this script does, so the Mac stays awake exactly as long as the wait.
    caffeinate -is -w $$ &
    while true; do
        if reply=$(on_pod "if [ ! -f $LOGS/$name.pid ]; then s=unknown
            elif kill -0 -- -\$(cat $LOGS/$name.pid) 2>/dev/null; then s=running
            else s=ended; fi
            at=\$(cat $DEADLINE 2>/dev/null); code=\$(cat $LOGS/$name.exit 2>/dev/null)
            echo \"\$s \${at:-none} \${code:-none}\""); then
            misses=0
            read -r state at code <<<"$reply"
            case $state in
            ended)
                if [[ $code = none ]]; then
                    result="$name ended without an exit status: it was stopped, or killed with the pod's processes"
                else
                    result="$name exited with $code"
                fi
                if [[ -n $path ]] && ! (pull "$path"); then
                    die "$result, but $path couldn't be pulled: pull it, or wait again"
                fi
                [[ $code = 0 ]] || die "$result; scripts/pod.sh log $name shows how"
                echo "$result"
                return
                ;;
            unknown) die "$name was never started on this pod" ;;
            esac
            # The watchdog deletes the pod, and everything on it, at its deadline, which only a person moves.
            soon=$(deleting "$at")
            if [[ -n $soon ]]; then
                [[ -z $path ]] || (pull "$path") || true
                die "$soon, and $name is still running: pull what you need, move the deadline more than" \
                    "6 hours away with scripts/pod.sh deadline HOURS, and wait again"
            fi
            # A pull that fails is tried again next time.
            if [[ -n $path ]] && ((SECONDS - pulled >= ${POD_PULL_SECONDS:-1800})); then
                (pull "$path") || echo "pod.sh: couldn't pull $path this time; trying again" >&2
                pulled=$SECONDS
            fi
        else
            # The pod's SSH sometimes drops a connection, so only a run of failures means it's gone.
            misses=$((misses + 1))
            ((misses < 5)) || die "the pod hasn't answered $misses times in a row: stopped waiting"
        fi
        sleep "${POD_WAIT_INTERVAL:-60}"
    done
}

# Each command runs on a line of its own, not after && or ||, so set -e stops it at its first failure.
command=${1:-}
need "$# >= 1" "args|setup|code|push|pull|start|log|stop|wait|deadline|ssh ..."
shift
case $command in
args)
    need "$# == 1" "args HOURS"
    args "$1"
    ;;
setup)
    need "$# == 0" "setup"
    connect
    setup
    ;;
code)
    need "$# == 0" "code"
    connect
    code
    ;;
push)
    need "$# == 1 || $# == 2" "push SRC [DEST]"
    connect
    push "$@"
    ;;
pull)
    need "$# == 1 || $# == 2" "pull PATH [DEST]"
    connect
    pull "$@"
    ;;
start)
    need "$# >= 2" "start NAME CMD.."
    connect
    start "$@"
    ;;
log)
    need "$# == 1" "log NAME"
    named "$1"
    connect
    on_pod "tr '\r' '\n' <$LOGS/$1.log | tail -n 30"
    ;;
stop)
    need "$# == 1" "stop NAME"
    connect
    stop "$1"
    ;;
wait)
    need "$# == 1 || $# == 2" "wait NAME [PATH]"
    connect
    wait_for "$@"
    ;;
deadline)
    need "$# <= 1" "deadline [HOURS]"
    connect
    deadline "$@"
    ;;
ssh)
    connect
    on_pod "$@"
    ;;
*) die "unknown command $command" ;;
esac
