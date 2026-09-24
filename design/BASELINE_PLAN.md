# ACE-2 fair baseline plan

This plan defines which comparisons can support ACE-2 claims and which are only
context. No baseline result is claimed until the named flow is executed and
linked to raw evidence.

## Comparison classes

| Class | Purpose | Claim status |
| --- | --- | --- |
| Same-RTL cross-platform PPA | Reproduce ACE-2 RTL/configuration through SKY130, Nangate45, and ASAP7 OpenROAD-flow platforms | Required PPA evidence; Nangate45 is academic and ASAP7 is predictive |
| Fair open accelerator baseline | Compare ACE-2 against executable open accelerators on the same workload boundary where feasible | Required when evidence permits |
| Software/system baseline | Provide end-to-end context for CPU/GPU/runtime inference on matched prompts and quantization where feasible | Context unless measurement boundaries are identical |
| Commercial/market reference | Published TOPS, bandwidth, process, or product claims | Context only; never apples-to-apples evidence |

## Open hardware baselines

| Baseline | Why relevant | Fairness requirement |
| --- | --- | --- |
| Gemmini | Open systolic-array generator with Chipyard/RISC-V integration and published accelerator methodology | Compare only with documented Gemmini configuration, supported datatype path, memory boundary, host/runtime overhead, and same Qwen2.5-0.5B command workload or explicitly documented subset |
| NVDLA-small | Open inference accelerator with documented RTL and tool flows | Use only if the workload adaptation is executable; transformer/SFU gaps must be reported, not extrapolated away |
| VTA or another open accelerator | Useful only for a reproducible open compiler/runtime comparison | Include only if it can execute a documented subset with frozen quantization and memory accounting |

Gemmini is the primary public baseline because it is the most directly relevant
open accelerator family for matrix-heavy inference. If Gemmini cannot execute a
required W4A8 transformer path without nontrivial missing operators, the result
must be labeled as a supported-subset comparison rather than an end-to-end
Qwen2.5-0.5B comparison.

## Apples-to-apples policy

A fair baseline comparison must document all of the following:

1. Same model boundary: Qwen2.5-0.5B, batch 1, same prompt/decode traces, and
   same tokenizer/runtime boundary.
2. Same numerical baseline or an explicit incompatibility: W4A8 with the same
   quality target is preferred; any int8, W8A8, or FP path is not a direct W4A8
   comparison.
3. Same memory-accounting boundary: external-memory bandwidth, bytes/token,
   on-chip SRAM/macro treatment, and host transfer overhead must be stated.
4. Same claim type: RTL/PPA is compared with RTL/PPA, FPGA with FPGA, and
   software runtime with software/runtime. Market data is not mixed into fair
   open-baseline tables.
5. Same evidence standard: raw logs, configs, source hashes, command manifests,
   and measurement scripts are retained and linked.
6. Same technology labeling: SKY130 is the primary ACE-2 public PDK target;
   Nangate45 is academic; ASAP7 is predictive; none are presented as measured
   silicon.

If any requirement is missing, the comparison is still useful engineering
context but cannot be used as a fair baseline win/loss claim.

## Minimum baseline report fields

| Field | Required content |
| --- | --- |
| Baseline name and revision | Repository, commit hash, generator parameters, and patches |
| Toolchain | Simulator/synthesis/flow versions and platform |
| Workload manifest | Prompt/decode trace IDs, model config, quantization config, command stream |
| Memory model | Interface width, effective bandwidth, latency/backpressure model, host overhead |
| Area/resource model | Standard-cell area, SRAM/macro area, FPGA LUT/BRAM/DSP, or software host resources as applicable |
| Timing/frequency | Achieved clock and failing paths if below target |
| Power | Activity source, corner, voltage, and reported components where available |
| Performance | Tokens/s, cycles/token, bytes/token, stall breakdown where available |
| Quality | BF16 reference, quantized reference, and measured delta |
| Limitations | Unsupported operators, substituted datatypes, unmodeled runtime costs, or missing public-tool support |

## Market-reference separation

Commercial edge AI accelerators, GPUs, NPUs, and published TOPS/W numbers may be
listed only in a market-context section. They cannot be used to claim ACE-2 is
faster, smaller, lower-power, or more efficient unless the measurement protocol,
process, workload, quantization, memory boundary, and evidence standard match.

## Definition-stage baseline status

At definition freeze, no baseline has been executed. The accepted next action
for a later benchmark stage is to run Gemmini first if its public toolchain can
be configured for a documented Qwen2.5-0.5B W4A8 or closest-supported workload.
Unsupported operators or datatype gaps must be reported as limitations rather
than hidden behind extrapolated end-to-end results.
