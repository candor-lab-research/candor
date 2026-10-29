# Running on a pod

Models the Mac can't hold, and jobs over a couple of hours, run on a Runpod pod. `scripts/pod.sh`
does everything but create and delete the pod; its header lists every command.

```
scripts/pod.sh args 24                 # 1. the command to create the pod with
export CANDOR_POD=root@IP:PORT         # 2. the pod's direct SSH address
scripts/pod.sh setup                   #    install candor there
scripts/pod.sh start qwen candor search runs/qwen   # 3. run a command detached
scripts/pod.sh wait qwen runs/qwen     #    keep the Mac awake and pull results as it goes
scripts/pod.sh pull runs/qwen          # 4. bring the results back, then delete the pod
```

## 1. Create the pod

`scripts/pod.sh args HOURS` prints the container command to create the pod with (its `entrypoint`
and `cmd`): Runpod's own start, and a watchdog that deletes the pod HOURS from now (at most 48),
whatever happens on this side, even if the pod's own start fails.

Create it with:

| Setting | Value |
| :--- | :--- |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Port | `22/tcp`, SSH on |
| CUDA | 12.8 or newer |
| Container disk | enough for the run (below) |

**Disk.** About 7.5 bytes a parameter for the run and its first export, 0.6 for each more export
kept, and about 10 GB of tools: about 85 GB for Qwen3.5-9B, 260 GB for a 27B with four exports kept.
A container disk can't grow without a restart, which wipes it, so `candor run` warns when an export
won't fit.

**GPUs and memory.** With several GPUs, the directions stage spreads the model across them, filling
each to a tenth less than it has free, and stops if any of it would go to the CPU or the disk. It
needs the model's weights in 16 bits and a tenth more across the GPUs together. Weights transformers
keeps off the GPUs, such as Qwen3.8-Flash-Next's 102 GB n-gram table, are joined from their shards in
the pod's memory, so `candor run` first checks the pod has twice their size and a tenth more.

## 2. Set it up

```
export CANDOR_POD=root@IP:PORT
scripts/pod.sh setup
```

`setup` checks the watchdog is armed, copies this working tree and your Hugging Face token, installs
candor, and checks that torch and llama-server both see the GPU. On Linux torch is the CUDA 12.8
build, as llama.cpp's release is.

`scripts/pod.sh code` copies the code again after a change and brings the pod's packages in line with
its `uv.lock`.

## 3. Run jobs

| Command | What it does |
| :--- | :--- |
| `push SRC [DEST]` | Copy a file or folder to the pod, each file with its time here to the nanosecond (below) |
| `start NAME candor ...` | Run a command detached, logged as NAME |
| `log NAME` | The end of that log |
| `stop NAME` | Stop that command |
| `wait NAME runs/RUN` | Wait for it to end, keeping this Mac awake |
| `pull PATH [DEST]` | Copy a file or folder back, leaving out GGUFs unless named (`pull runs/NAME/exports/trial-37`) |
| `ssh [CMD...]` | A shell or a command on the pod |

- Paths are the same as here, under `/root/candor` on the pod. Anything there that git doesn't ignore
  is replaced by the next `code`, which refuses while a command is running, as it would change the
  code under it.
- rsync from macOS keeps file times only to the second, and a local model's fingerprint holds them to
  the nanosecond, so `push` gives the pod's copy this Mac's times and checks every file's size and
  time against them. It fails, naming a file, where the pod's disk can't keep them, and names files
  a pushed folder holds on the pod but not here, which would change its fingerprint. Links are copied
  as the files they point to, and a link to nothing stops the push.
- `wait` keeps the Mac awake until the command ends, so nothing needs a timed `caffeinate`. It pulls
  `runs/RUN` every 30 minutes and once more at the end, and exits 0 only if the command did.
- `wait` stops early, with an error, if the pod stops answering for about five minutes or the
  watchdog's deadline is less than 6 hours away: move the deadline and wait again.

## 4. Finish

- `scripts/pod.sh deadline` checks the watchdog and says when it deletes the pod.
  `deadline HOURS` moves that to HOURS from now.
- Everything lives on the container disk, which Runpod wipes when a pod is stopped or restarted, and
  a stopped pod's watchdog can't delete it. So pull the results, then **delete the pod, never stop
  it**.
