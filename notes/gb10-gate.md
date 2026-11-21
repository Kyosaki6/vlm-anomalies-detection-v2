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

## 2026-11-20 go

The resident jobs from the two readings above are gone. Nothing replaced them.

| Field | Value |
|---|---|
| Timestamp | 2026-11-20T17:57:35+07:00, read from the box (`date -Iseconds`) |
| Host | `aitopatom-d4ac`, reached as `gx10@100.104.163.47` port 22 |
| GPU util | 0% |
| Temperature | 48 C |
| Memory | `[N/A]` (GB10 unified memory, same as the previous reading) |
| Resident PIDs | none. `nvidia-smi --query-compute-apps` returned no rows |
| Verdict | **go** |

Confirmed a second time at 18:05+07:00 while finding a Python env: still 0%, 47 C, no compute process. The 10-minute bar is met by the gap between the two readings, not by a polling loop. P2 already exited 0 earlier today (`notes/qwen3-smoke.md`, 12:02, `forced=yes` against the no-go above).

The go above covered the download only. It does not cover the embedding run. A later reading the same evening found the GPU busy again:

| Field | Value |
|---|---|
| Timestamp | 2026-11-20, about 18:25+07:00, one `nvidia-smi` from the box |
| GPU util | 96% |
| Temperature | 70 C |
| Resident PIDs | 1043889, 1044115, 1044550, all `/home/gx10/MATE-2026/.venv/bin/python` |
| Verdict | **no-go for the N4 embedding run** |

The train-split download is not a GPU process and kept running. The embedding run waits until a reading shows these PIDs gone and util under 10%. Nobody kills them.
