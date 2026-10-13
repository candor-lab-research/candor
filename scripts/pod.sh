#!/bin/bash
# Runs candor on a Runpod pod from this Mac. The pod is created and deleted with Runpod's own tools;
# this arms its watchdog, sets it up and moves files to and from it. Run it from the repository root.
#
#   scripts/pod.sh args HOURS        the container command to create the pod with: Runpod's own
#                                    start, and a watchdog that deletes the pod HOURS from now
#   scripts/pod.sh setup             check the watchdog is armed, copy the code and the Hugging Face
#                                    token, install candor, and check torch and llama-server see the GPU
#   scripts/pod.sh code              copy this working tree's code again, uncommitted changes too
#   scripts/pod.sh push SRC [DEST]   copy a file or folder to the pod (DEST defaults to SRC)
#   scripts/pod.sh pull PATH [DEST]  copy a file or folder back (DEST defaults to PATH), leaving
#                                    out gguf/ and exports/ folders unless PATH is one of them
#   scripts/pod.sh start NAME CMD..  run a command in the code folder, detached, logged as NAME
#   scripts/pod.sh log NAME          the end of that log
#   scripts/pod.sh stop NAME         stop that command
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
    ssh_command="ssh -p $port -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS -o ServerAliveInterval=30"
}

on_pod() {
    $ssh_command "$host" "$@"
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
    local at
    at=$(on_pod "cat $DEADLINE 2>/dev/null") ||
        die "the pod has no deadline: it wasn't created with scripts/pod.sh args, so delete it yourself"
    # The whole command line, as the container's own shell and this check's both mention the file.
    on_pod "pgrep -fx 'bash /pod-watchdog.sh' >/dev/null" || die "the watchdog isn't running: delete the pod yourself"
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

code() {
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
    code
    on_pod "umask 077 && mkdir -p ~/.cache/huggingface && cat >~/.cache/huggingface/token" <"$TOKEN"
    on_pod "[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh >/dev/null"
    on_pod "cd $CODE && ~/.local/bin/uv sync --frozen --quiet"
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
    # setsid makes the command lead its own process group, which stop ends as a whole.
    on_pod "set -e; mkdir -p $LOGS; cd $CODE
        if [ -f $LOGS/$name.pid ] && kill -0 -- -\$(cat $LOGS/$name.pid) 2>/dev/null; then
            echo '$name is still running: stop it first' >&2
            exit 1
        fi
        PATH=$CODE/.venv/bin:\$PATH setsid nohup $(printf '%q ' "$@") >$LOGS/$name.log 2>&1 </dev/null &
        echo \$! >$LOGS/$name.pid"
    echo "started; scripts/pod.sh log $name"
}

stop() {
    named "$1"
    on_pod "kill -TERM -- -\$(cat $LOGS/$1.pid)" || die "$1 isn't running"
    echo "stopped $1"
}

# Each command runs on a line of its own, not after && or ||, so set -e stops it at its first failure.
command=${1:-}
need "$# >= 1" "args|setup|code|push|pull|start|log|stop|deadline|ssh ..."
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
