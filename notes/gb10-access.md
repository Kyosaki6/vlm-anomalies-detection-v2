# GB10 access (P3)

How a second person reruns the smoke in `notes/qwen3-smoke.md` without asking anyone. No key material lives in this file. If a command below fails, the failure is the answer: do not guess a flag.

## Host

| Field | Value |
|---|---|
| Address | `100.104.163.47` (Tailscale) |
| SSH user | `gx10` |
| Port | `22` |
| GPU | NVIDIA GB10, driver 580.178.04, CUDA 13.0 |
| Hostname on the box | `aitopatom-d4ac` |

Login is by key already installed on the caller's machine. This note does not carry a key, a password, or a `ssh` line to copy: write the command yourself from the table so it never lands in shell history inside the repo.

## Who holds a key

| Person | Key | May launch a GPU job |
|---|---|---|
| Thịnh | yes, the account holder | yes. Every launch goes through him while he holds the box |
| Phước | no | no. Owns the smoke code; Thịnh runs it |
| Nam | no | no. Owns the embedding job; Thịnh runs it |
| Thanh | no | no. CPU only this phase |

A person without a row does not get a key added here. Ask, then edit the table in the same commit as the key install.

## A job dies with the ssh session

The smoke and the embedding run are plain `python` over ssh, not systemd and not `nohup`. Closing the session sends SIGHUP and the process dies with it. That is the intended behaviour: a detached job would outlive the person watching the GPU, and the rule is one process at a time.

To keep a job across a dropped connection, run it inside `tmux` on the box and reattach. Do not background it with `&` and disconnect. A job left running after the ssh session closes is a resident process the next gate reading has to treat as occupied, and it blocks P2 and N4.

## Before launching

1. Read `notes/gb10-gate.md`. The newest section must say `go` and be dated today. A `no-go`, or a `go` from another day, means stop.
2. `nvidia-smi` must show GPU-util under 10% and no compute process you do not own. Never kill a resident PID. If one is there, the gate was wrong: fix the note, do not the process.

## Smoke command

Runs from the repo root on the box, after the gate says go. Writes `notes/qwen3-smoke.md` and exits 0 on success. An OOM is recorded in that note and ends Phase 2; there is no 4-bit fallback.

```bash
python scripts/qwen3_smoke.py --quantization 8bit --out notes/qwen3-smoke.md
```

`scripts/qwen3_smoke.py` is P2 and does not exist until that task lands. Until then this command fails with `No such file`, which is correct.
