# GB10 gate (P0)

Read-only check before any GPU launch. A go is valid only on the day it was written (binding B7). A no-go is a finished task: steps 1-3 of Phase 2 still land, P2 and N4 wait. Nobody polls the GPU in a loop.

## 2026-11-19 no-go

| Field | Value |
|---|---|
| Timestamp | 2026-11-19T20:40+07:00 |
| From | this workstation, no Tailscale route of its own |
| Command | `ssh -o BatchMode=yes -o ConnectTimeout=8 gx10@100.104.163.47` |
| Result | `ssh: connect to host 100.104.163.47 port 22: Connection timed out` |
| GPU util | not read. The host did not answer. |
| Temperature | not read. |
| Resident PIDs | not read. |
| Verdict | **no-go** |

Last reading that did land, from the plan (`notes/2026-11-19-plan-qwen3-vl-lora-raft-fork.md`, 2026-11-19 20:10 local, via Tailscale on `gx10@aitopatom-d4ac`): util 96%, 69 C, P0, resident PID 401001 `python` at 17024 MiB plus PID 535418 and PID 537304 `MATE-2026/.venv/bin/python` at 2428 MiB each. That snapshot is stale and was already a no-go on its own (util far above the 10% bar). It is cited here so the next reading has something to compare against. It is not a go.

P2 (8-bit smoke) and N4 (embed the 1610) do not start off this note. Re-run the reading from a machine that can reach the box, write a new section below this one, and only a same-day `go` unblocks them.

## 2026-11-19 no-go, second reading

The host answers this time. The GPU does not.

| Field | Value |
|---|---|
| Timestamp | 2026-11-19T21:25:19+07:00, read from the box (`date -Iseconds`) |
| Host | `aitopatom-d4ac`, reached as `gx10@100.104.163.47` port 22 |
| GPU util | 95% |
| Temperature | 71 C |
| Memory | `[N/A]` from the query (GB10 unified memory does not report `memory.used` this way) |
| Resident PIDs | 548193 `python` 17026 MiB; 572109 `MATE-2026/.venv/bin/python` 2424 MiB; 572949 `MATE-2026/.venv/bin/python` 2424 MiB |
| Verdict | **no-go** |

95% is not under 10%, and three compute processes are resident. Nobody kills them. P2 and N4 stay blocked until a later reading the same day shows util under 10% for 10 continuous minutes and these PIDs gone. No polling loop from here.
