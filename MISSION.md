# ACE-2: ARGUS Compute Engine

## Objective

Design a complete, computer-verifiable pre-tapeout digital ASIC accelerator for
Qwen2.5-0.5B batch-1 edge inference. This is a standing engineering campaign,
not a one-shot RTL generation task and not a claim of fabrication.

## Frozen product contract

- Model: Qwen2.5-0.5B.
- Numerical baseline: W4A8 (4-bit weights, 8-bit activations) with explicitly
  justified accumulator, scaling, rounding, saturation, and KV-cache formats.
- Primary technology: SKY130 using public, reproducible libraries and tools.
- Primary timing floor: at least 100 MHz.
- Primary area cap: at most 2.0 mm^2 of non-SRAM standard-cell area.
- Memory boundary: an abstract, backpressured streaming external-memory
  interface. Sweep bandwidth rather than binding the first version to a PHY.
- Performance objective: maximize sustained end-to-end tokens/s subject to
  correctness, quality, frequency, and area constraints.
- Comparative PPA: reproduce the same RTL/configuration through supported
  Nangate45 and ASAP7 OpenROAD-flow platforms. Label 45 nm as an academic
  platform and ASAP7 as predictive; never present either as measured silicon.
- Public baseline: run a fair, configuration-documented comparison with Gemmini
  and other directly relevant open accelerators when evidence permits.
- Delivery level: `pre_tapeout`. Complete every computer-executable digital
  design check that available public tools support, without claiming a foundry
  submission or fabricated chip.

## Required system scope

Deliver a coherent accelerator subsystem rather than an isolated MAC:

- host-visible control/status and interrupt behavior;
- command/descriptor engine;
- external-memory streaming/DMA boundary;
- banked on-chip SRAM interfaces and explicit macro/blackbox accounting;
- reusable compute resources for projections, attention, and MLP;
- normalization, RoPE, KV-cache, softmax, activation, residual, and
  requantization paths needed by the frozen model boundary;
- independent executable reference model;
- cycle-accurate RTL verification;
- compiler/runtime or deterministic command-stream generation sufficient to
  execute the frozen workload;
- reproducible synthesis, DFT, physical design, and sign-off evidence.

## Area-reuse mandate

The prior ACE effort exceeded the 2.0 mm^2 non-SRAM cap. Reuse is therefore a
first-class architecture constraint:

- evaluate folding/sharing of MAC lanes, accumulators, requantization,
  vector/SFU operations, divider, round/saturate logic, DMA, buffers, and
  mutually exclusive state;
- quantify mux/control and cycle overhead before accepting a shared design;
- keep an append-only frontier ledger with RTL/constraint hashes, delta cells,
  delta non-SRAM area, Fmax, cycle/throughput impact, and remaining reserve;
- stop wording-only or repeated low-yield local tweaks; escalate to structural
  folding, a different dataflow, or an explicit Pareto/no-go result;
- never relax the 100 MHz or 2.0 mm^2 contracts without operator approval.

## Fast capability and PPA decision policy

After the one-time definition, architecture, and environment bootstrap, ACE-2
uses a fast bounded loop for non-milestone operator uplift:

`RTL implementation -> full verification -> fresh canonical SKY130 PPA -> evidence binding`

Intermediate operator additions are not release milestones. They do not reopen
definition, architecture, environment, prototype, full benchmark, or signoff
unless the frozen contract changes or a complete decoder/model milestone is
reached.

Every fresh PPA frontier must record the ordered supported layer/operator
prefix, first unsupported layer/operator, mode, decision, RTL and constraint
hashes, cells, non-SRAM area, Fmax, cycle or tokens/s impact, delta area
percent, compression improvement percent, and remaining 2.0 mm^2 / 100 MHz
reserve.

The operator-owned decision rules after each fresh PPA are:

1. If total non-SRAM area exceeds 2.0 mm^2, enter `GLOBAL_COMPRESSION`.
   Compression must consider lifetime-based sharing/folding across MACs,
   accumulators, requantization, vector/SFU, divider, round/saturate, DMA,
   buffers, state, and control; local micro-optimization of only the newest
   block is not sufficient.
2. In `GLOBAL_COMPRESSION`, continue the same compression direction only when
   the latest verified change improves non-SRAM area by at least 3% while
   preserving correctness and at least 100 MHz. Below 3%, stop that direction
   and resume capability advancement or choose a structurally different global
   mechanism.
3. In `ADVANCE`, continue adding the next layer/operator when the verified
   non-SRAM area increment is less than 3%. If the increment is at least 3%,
   pause advancement for global reuse/folding review and compression before
   adding another capability.
4. Every frontier records the current supported prefix, first unsupported
   item, mode, decision, RTL and constraint hashes, cells, non-SRAM area, Fmax,
   cycle or tokens/s impact, delta area percent, compression improvement
   percent, and remaining area and frequency reserve.
5. Intermediate operator additions are not release milestones. Do not rerun
   definition, architecture, environment, prototype, full benchmark, or
   signoff unless their contract changed or a complete decoder/model milestone
   is reached.
6. Framework self-maintenance must not displace chip progress unless a concrete
   framework defect blocks execution. Finish a bounded blocking repair, then
   return immediately to ACE-2 architecture and the fast RTL loop.

The 2.0 mm^2 non-SRAM cap and 100 MHz floor remain unmet until evidence
satisfies them. Planner-authored reports may recommend changes but cannot
approve replacement targets.

### Sealed QECR no-go and certified harness gate

`shared_down_projection_residual_fusion_v1` is sealed by the independently
accepted integrity no-go at
`evidence/shared_down_projection_residual_fusion_v1/latest/VERIFICATION_DECISION.json`
(SHA-256 `9655ca8a724fc04c6e32c26f7883f07d3ff117b82d2aa04ebbc68b15de26d8c9`).
Its candidate capability was not accepted and no paired smoke, shell admission,
candidate PPA, physical design, prototype, benchmark, signoff, tapeout, or
silicon run followed.

`cross_layer_quantization_error_carry_final_output_v1` is also sealed after the
terminal baseline protocol/provenance no-go and Manager recovery seal. Its
baseline and candidate may not be rerun, repaired into a pass, or used as a
successor. Accidental post-terminal RTL/precheck artifacts are retained only as
negative provenance and are not accepted implementation evidence.

The candidate-independent synthetic/dry-fixture baseline harness is independently
certified for atomic artifact-plus-companion publication, exact provenance,
durable exactly-once reservation, architecture-stage gating, candidate-entrypoint
interlock, crash recovery, and fail-closed artifact validation. This certification
does not run a model or candidate and does not freeze a successor. The Manager must
freeze a structurally distinct successor before the architecture checklist can
close or any downstream work can begin.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; `ADVANCE` mode, the 2.0 mm^2 cap, the 100 MHz floor, the
128-bit memory boundary, and the historical PPA frontier remain unchanged.

## Verification and implementation contract

Progress through the `chip_design` vertical with independent Reviewer approval.
At minimum:

1. Freeze workload, interface, quality, and acceptance criteria.
2. Build an independent reference/oracle and representative model traces.
3. Implement synthesizable SystemVerilog with explicit CDC/RDC/reset behavior.
4. Run lint, deterministic and randomized simulation, assertions/formal where
   tractable, coverage, and equivalence checks.
5. Add DFT/scan architecture and execute available scan/ATPG checks.
6. Run synthesis and technology mapping with source/target/constraint hashes.
7. Run floorplan, PDN, placement, CTS, routing, extraction, STA, and power.
8. Run available SI, IR-drop, EM, DRC, LVS, and antenna checks.
9. Produce SKY130, Nangate45, and ASAP7 PPA reports with honest platform labels.
10. Compare Gemmini on matched workload, quantization/quality, memory bandwidth,
    host boundary, resource/area budget, and measurement protocol.

All PPA and benchmark claims must link to immutable raw output. Missing public
tool support is a visible blocker or limitation, never a success-shaped default.

## Recovery boundary

ACE-1's old server is unavailable. Do not claim its RTL was recovered. Use
`RECOVERY_BASELINE.md` only as historical design evidence and reimplement every
source file whose exact content and hash are unavailable. New ACE-2 evidence
starts from this repository's first commit.

## Public observability

Maintain privacy-filtered machine-readable status for
`https://ace.argusbot.cn/process-dashboard.html`. It may expose stages,
Reviewer-certified metrics, blockers, timestamps, and public artifact hashes.
It must not expose prompts, credentials, private paths, raw private logs,
private PDK contents, or unsupported claims.
