# ACE-2 host-only dependencies

This repository is a curated source snapshot, not a full host image.

| Dependency | Location / provenance (updated September 26) | Published here | Recovery consequence |
| --- | --- | --- | --- |
| Live Argus state, backlog, handoffs, native events, claims | `/home/argustest/.argus-skill-ace2/projects/s-c8ae985b` | No | Same-host continuation can inspect it. Disk loss removes lifecycle and authoritative receipts not separately archived. |
| Active runtime source | `managed-maintenance/ace2-host-consumer-bridge-20260913/candidate`; runtime digest `7ebd4c86...db1227a` | No | Reconstruct/replace Argus runtime separately and revalidate import identity. Do not launch from this repository by assumption. |
| Accepted diagnostic result | `reports/verification/persistent-conversation-active-rtl-diagnostic-0005/result.json`; SHA-256 `5795ee56...cc4f893` | No, hash and summary only | The public summary is not the original receipt. |
| Running task state | mission `04bdbc62467f`, namespace `0008` at the September 26 observation | No | No final three-turn qualification is claimed. Never restore old PID/lock/claim records as active state. Inspect the original host. |
| Full model weights, adapters, checkpoints, tokenizer/cache | Host-local model/checkpoint/cache locations, governed by their upstream licenses | No | Reacquire from approved provenance and verify exact hashes/licensing before use. |
| Raw reports, simulation output, VCD, generated bulk | Live repository `reports/`, raw benchmark/verification trees, build/output trees; most of the roughly 48 GiB non-build workspace payload | No | Recompute only under a newly authorized task; do not claim old acceptance from regenerated output. |
| GitHub credential | User-supplied `GH_CONFIG_DIR=/home/argustest/argustest2/.gh-config` | Never | Supply externally; expected GitHub login is `aHappend`. |
| Copilot profile/credentials | `/home/argustest/.copilot` | Never | Supply/restore locally; it is independent of GitHub identity. |
| Scheduled supervision | External/session-scoped registration | No | Re-establish explicitly. Repository presence does not prove a schedule is active. |

Small source fixtures and the large generated SystemVerilog compilation source
`verification/generated/projection_vectors.svh` are included and hashed in the
source manifest. Raw experimental payloads and credential-bearing state are not.
