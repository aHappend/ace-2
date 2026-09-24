# ACE-2 RTL traceability notes

## Layer19 Provenance Repair V3 (2026-09-02)

`design/RTL_ACTIVE_CLOSURE_PROVENANCE.json` is the canonical path-qualified
provenance ledger for all 18 files in the active RTL compilation/include
closure. Each record binds the unchanged source SHA-256 and explicitly states
first-party origin, ACE-2 project stewardship, third-party status, and the
applicable license/disposition. The repository does not declare a
repository-wide external license, so the records make the narrower honest
disposition: operator-authorized project-internal use only, with no external
redistribution grant claimed.

The four generated-source records additionally bind the generator path and
SHA-256 version, exact regeneration command and working directory, repository
input paths and SHA-256 values, and target output SHA-256 for the Layer-23
rank-1 payload, model-parameter header, SiLU input-scale table, and SiLU LUT.
The Layer-23 generator's two operator-managed Fresh-L2 receipt prerequisites
are identified and digest-bound without restating their private locations.
Regeneration is documented but was not run. The source count remains 18 and
the closure manifest remains
`54a9b34e502928eaf7515a64f167ce00e2206624a1de1dcf6ad6434fe5419fbc`.
No `.sv` or `.svh` byte, RTL behavior, checkpoint, evidence, authority, or
certified result changed, and no model, reference, RTL, unit, token, PPA,
FPGA, or U280 workload was run.

The manifest now marks its broad legacy `ip_provenance` array as a historical
mixed inventory rather than the active-closure authority, and refreshes the
legacy current SiLU LUT generator/reference hashes to remove a contradictory
stale claim. The exact 18-source authority is
`design/RTL_ACTIVE_CLOSURE_PROVENANCE.json`.

The fresh successor namespace is
`reports/ace2-layer19-rtl-admission-0004`; its paired review namespace is
`reports/ace2-layer19-rtl-admission-review-0004`. These namespaces do not
admit Layer 19 or close any checklist predicate. Mandatory framework L2 must
independently and explicitly decide `rtl.contract-traceability`,
`rtl.hardware-discipline`, and `rtl.ip-provenance` against the sealed
successor dossier.

## Current RTL binding for Manager-approved read-only audit (2026-08-29)

This section is the live-tree traceability note for the explicitly
Manager-approved read-only current-tree audit. It does not change the
Manager-owned project stage, which remains `stage1_software_diagnosis`. Later
sections are historical publications and rejected candidates; they do not
redefine the current productization contract.

The exact Makefile compilation-input closure plus recursively resolved local
includes contains 18 `.sv`/`.svh` files at SHA-256
`54a9b34e502928eaf7515a64f167ce00e2206624a1de1dcf6ad6434fe5419fbc`.
The current `rtl/ace2_shell.sv` hashes to
`77eadf2546f36672a3b76c67ab536e7962de5fbed1069804befc7f96627281bc`;
`rtl/ace2_pkg.sv` hashes to
`80916442d452056eb49e52c48f955fbf9a1205610322ad36c5ea552556820c28`;
`rtl/ace2_layer23_v_rank1_integer_correction_sidecar.sv` hashes to
`5851d5fe3d9473ae38837dc27b668d7cd2b403dd7f52256bdab2eeca381be5f8`;
and its generated payload include hashes to
`145a8d57b8a92e55390f912dac14e7b72ce2208300fed979d8376b60744ebd85`.
The compilation set is broader than the elaborated shell hierarchy; rejected
diagnostic modules are parsed but are not instantiated product logic.

Earlier accepting audits are historical because their documentation and public
shell bindings precede this 14-parameter closure. The fresh audit is structural
and compile-only; it runs no simulation, formal, synthesis, timing, PPA,
benchmark, model/runtime workload, FPGA, or U280 flow.

| Live state/datapath/interface family | Frozen specification authority | Synthesizable implementation |
| --- | --- | --- |
| Public parameters and ports | `design/SPEC.md` “Public ace2_shell parameter and port contract” | `rtl/ace2_shell.sv` module header; the audit checks all 14 parameters and 64 ordered ports, directions, evaluated widths, and unsigned public declarations. `ENABLE_LAYER23_V_RANK1_SIDECAR` qualifies `layer23_v_rank1_command_match_buf_w` and maps to sidecar `ENABLE_SIDECAR`; `LAYER23_V_RANK1_CONFIG_VALID` gates `layer23_v_rank1_command_allowed_w` fail closed and maps to sidecar `CONFIG_VALID`. |
| Command buffering, decode, active opcode/shape/address state, and serialized acceptance | `design/SPEC.md` “Stage 1 cycle-level prefill, KV, and decode behavior” and “Direct-command control and legal-value rules” | Shell command buffers, `ST_IDLE`, `ST_CMD_DISPATCH`, `descriptor_valid_q`, `op_kind_q`, and command-field registers |
| External-memory and completion ready/valid bookkeeping | `design/SPEC.md` “Clock, reset, flow-control, latency, and throughput rules” and the Stage 1 KV/completion rows | `read_outstanding_q`, `write_data_pending_q`, `write_response_outstanding_q`, registered request payloads, `ST_WRITE_RESP`, `ST_WRITE_RETIRE`, and completion hold state |
| RMSNorm sum, inverse RMS, output beats, and completion fields | Active opcode `0x02`, reset-observable values, and public completion-port contracts; preserved predecessor “Fixed-point and scaling contract” | `ace2_rmsnorm_core`, shell `ST_START` through `ST_WAIT_DONE`, and the repaired `done_sumsq_q`/`done_inv_rms_q30_q` capture; non-RMS or invalid completions drive these outputs to zero |
| Projection, RoPE, score, softmax, attention value/compose, residual, SiLU, and KV datapaths | Active opcode table and Stage 1 operator grammar; preserved predecessor “Exact per-layer execution and carry schedule”, “Exact RTL interface composition”, and “Fixed-point and scaling contract” | The corresponding active shell state clusters and elaborated cores listed by the hierarchy audit |
| Dynamic Scale32 layer-0 RMS/Q/K/V path | Active opcode table permits flag bit 6 only for authenticated layer-0 tuples; illegal descriptors and mismatched responses fail closed | `ST_DYN_SIDECAR_*`, delta-application state, and `ace2_dynamic_scale32_sidecar_validator_core`; these states implement the authentication mechanism without changing the public interface |
| Layer-23 V rank-1 sidecar width, mux, gating, and reset isolation | Exact 896-element signed-int8 V input and 128-element signed-int8 correction output; V-only operation; eight ordered 16-lane output beats; default-disabled and configuration-valid parameter gates; hard/soft-reset isolation | `rtl/ace2_layer23_v_rank1_integer_correction_sidecar.sv`, its generated payload include, and the shell `layer23_v_rank1_*` command/mux state. `ENABLE_LAYER23_V_RANK1_SIDECAR=0` preserves the baseline path, `LAYER23_V_RANK1_CONFIG_VALID=0` fails the guarded command before memory traffic, only the Layer-23 V descriptor selects corrected data, and hard or soft reset clears provisional sidecar state without leaking corrected output into another command. |
| Reset, soft reset, watchdog, error, interrupt, and provisional-state invalidation | Active clock/reset/flow-control rules, reset-observable values, and reset/illegal acceptance rows | Single `clk_i` domain, asynchronous active-low reset assertion, clocked state release, `state_soft_reset_q`, watchdog/error registers, and response-fault paths |
| Frozen arithmetic widths and table-backed constants | Active public parameter tuple plus preserved predecessor fixed-point ranges | `rtl/ace2_pkg.sv`, core parameters, explicit signed casts, generated SiLU/softmax tables, and width-clean shell descriptor counters |

The canonical “Current implemented CSR protocol and register map” section maps
one-to-one to the following live logic. These compatibility registers do not
create an autonomous descriptor-ring execution path.

| CSR/interface response | RTL state and exact effect |
| --- | --- |
| Request/response protocol and address errors | `csr_req_valid_q`, `csr_req_*_q`, `csr_ready_o`, and the registered response block around `csr_read_value`/`csr_known_addr`; `csr_rvalid_o`, `csr_rdata_o`, and repaired `csr_error_o` hold until `csr_rready_i` |
| `ID` `0x000` | `ACE2_ID_VALUE`; constant read, writes ignored |
| `VERSION` `0x008` | `ACE2_VERSION_VALUE`; constant read, writes ignored |
| `CAPABILITIES` `0x010` | `ACE2_CAP_VALUE`; constant read, writes ignored |
| `CONTROL` `0x018` | `control_q[4:0]`, `control_enable_w`, `soft_reset_req_w`, `irq_global_enable_w`, `strict_errors_w`, and `perf_clear_w`; low-byte strobe only and pulse bits self-clear |
| `STATUS` `0x020` | `csr_read_value` constructs bits from `shell_busy_w`, `halted_on_error_w`, `!cmd_valid_i`, and `watchdog_active_w` |
| `ERROR_STATUS` `0x028` | `error_status_q[5:0]`; W1C low byte plus descriptor, response-memory, numeric, watchdog, reset-busy, and strict unknown-CSR set sites in the shell state machine |
| `INTERRUPT_ENABLE` `0x030` | `interrupt_enable_q[5:0]`; low-byte-strobed RW mask |
| `INTERRUPT_STATUS` `0x038` and `irq_o` | `interrupt_status_q[5:0]` W1C; completion/descriptor on bit 0, memory bit 1, numeric bit 2, watchdog bit 3, reset-busy bit 4; `irq_o=control_q[2] && |(interrupt_status_q & interrupt_enable_q)` |
| `DESC_BASE` `0x040` | `desc_base_q` byte-strobed storage/readback only; no dispatch consumer |
| `DESC_HEAD` `0x048` | `desc_head_q[31:0]`, explicit `apply_wstrb32`, zero-extended readback, and modulo-2^32 updates |
| `DESC_TAIL` `0x050` | `desc_tail_q[31:0]` plus `desc_tail_inc_q`; zero-extended readback and one-cycle-pipelined qualifying retirement increments |
| `DOORBELL` `0x058` | Known zero read; any write performs `desc_head_q <= desc_head_q + 32'd1`; no descriptor fetch or command dispatch |
| `WATCHDOG_LIMIT` `0x060` | `watchdog_limit_q`, `watchdog_count_q`, `watchdog_armed_q`, `watchdog_fire_q`, forward-progress reload, and watchdog terminal-error path |
| `PERF_CYCLE` `0x068` | `perf_cycle_event_q` and `perf_cycle_q`; registered enable-cycle events and wrapping increment |
| `PERF_BYTE` `0x070` | `perf_byte_event_q` and `perf_byte_q`; one 16-byte increment for an accepted read response or write-data event |
| `PERF_TOKEN` `0x078` | Dedicated `perf_token_q` process; low-32 projection-`m` or unit retirement increments, upper half preserved |
| `PERF_STALL` `0x080` | `mem_stall_w`, `perf_stall_event_q`, and `perf_stall_q`; one increment for any registered external stall cycle |
| Hard reset and soft reset CSR effects | Hard-reset assignments in the main CSR/state process and dedicated token/tail processes; `soft_reset_req_w` clears provisional/completion and sticky cause state, sets reset-busy when applicable, and does not reset the documented configuration/counters unless `perf_clear_w` is also active |

The canonical “Active layer-0 Dynamic Scale32 contract” maps to the exact live
mechanism below. The standalone 64/128-lane group builder and tagged
accumulator remain broader reusable modules; only this table is instantiated
in the shell command path.

| Dynamic Scale32 requirement | RTL implementation |
| --- | --- |
| Four exact flag-`0x49` tuples | `DYNAMIC_SCALE32_FROZEN_FLAGS`, fixed RMS/Q/K/V address constants, `rms_dynamic_initial_buf_w`, `proj_dynamic_q_buf_w`, `proj_dynamic_k_buf_w`, `proj_dynamic_v_buf_w`, and descriptor-valid gating |
| Sidecar location and four-beat fetch | `prefix_mem_req_addr_q <= src0_addr_q-64`, `ST_DYN_SIDECAR_REQ`, `ST_DYN_SIDECAR_RECV`, `dynamic_sidecar_beat_q`, and `dynamic_sidecar_q[511:0]` |
| Fixed live shape and base scales | Validator instance uses group lanes 128, group count 7, tensor elements 896, `MAX_GROUPS=38`, and `dynamic_initial_base_scales_w={38{32'h00008000}}` |
| Producer identity | `dynamic_expected_producer_tag_w`, `dynamic_expected_layer_w`, and `dynamic_expected_opcode_w`; RMS uses sequence position/layer `0xff`/opcode `0xfe`, while Q/K/V use modulo-2^16 completion-tag offsets, layer 0, and opcode `0x02` |
| Byte-exact `BFP1` validation | `ace2_dynamic_scale32_sidecar_validator_core` checks magic/schema, lanes/count/zero byte, producer identity, element count, model identity `0xadcc20785542c188`, bytes 62:63, unused delta bytes, payload alignment/lower bound, and base Scale32 normalization |
| Delta/effective exponent legality | Validator reads signed bytes 24..61, requires delta `[-24,+24]`, and requires base exponent plus delta `[-24,+4]`; fixed base exponent zero narrows the active seven deltas to `[-24,+4]` |
| Projection dynamic-to-base application | `dynamic_proj_sidecar_group_w=proj_group_idx_q[7:5]`, `dynamic_apply_delta_s8`, per-lane overflow flags, and `dynamic_proj_apply_overflow_q`; positive delta shifts left, negative delta right-shifts ties-to-even, range `[-128,127]` |
| RMSNorm base-to-dynamic output | `dynamic_output_group_w=out_idx_q[5:3]`, `dynamic_quantize_delta_s8`, `dynamic_output_overflow_lane_w`, and `dynamic_output_numeric_overflow_q`; negative delta shifts left, positive delta right-shifts ties-to-even, range `[-127,127]` |
| Fail-closed error classification | `ST_DYN_VALIDATE_WAIT` maps structural validation to descriptor error and exponent failure to numeric error; projection-apply overflow and RMS output overflow set numeric error and terminal command error; no successful completion follows |
| Generated RMSNorm output sidecar | `dynamic_output_sidecar_w` fixes magic/schema/128 lanes/7 groups/tag/layer/opcode/elements/model identity, copies input delta bytes 24..30, zero-fills the remainder, and writes four beats at `dst_addr_q-64` through `ST_DYN_OUTPUT_WRITE_REQ/DATA` with memory tag `0x13` |
| Reset invalidation | Hard reset and `soft_reset_req_w` clear `dynamic_sidecar_q`, `dynamic_sidecar_beat_q`, `dynamic_output_numeric_overflow_q`, and `dynamic_proj_apply_overflow_q` |

The cycle-1 repair removes orphan provisional CSR/completion declarations,
declares the descriptor-tail increment state, makes descriptor head/tail
extension and updates explicitly 32-bit, and latches RMSNorm completion values
before exposing them under completion backpressure. It does not alter the
14-parameter/64-port public contract.

Cycle 7 resolves the remaining traceability issue from current binding facts.
Canonical `design/SPEC.md` and `design/BENCHMARK_INTERFACE.json` contain no
`DESC_SIZE`, `LAST_RESULT`, or `LAST_ERROR_INFO` definition, and the Manager
explicitly corrected the historical transition that named those CSRs. The
three unused package constants and three unused shell imports were therefore
removed. No state, decode, address, or behavior was added or changed. The
contract audit now rejects either silent reintroduction of those RTL symbols
or an unreviewed canonical-contract appearance of the historical names.

Icarus elaboration, Verilator shell lint, hierarchy extraction, all four
Dynamic Scale32 top lints, and repository-inventory lint pass in
`build/rtl-stage-audit-20260813-planner-cycle1-repair/`. Active logic has no
`WIDTH`, `LATCH`, `MULTIDRIVEN`, `UNDRIVEN`, `UNOPTFLAT`, `CASEINCOMPLETE`,
`SYNCASYNCNET`, or unsupported-design-path finding. Repository-only width and
reset-use warnings remain confined to the independently rejected,
uninstantiated `ace2_dynamic_rope_head_core`.

Fresh review supports synthesizable discipline, width/reset/parameter intent,
and the CSR repair, but found the active Dynamic Scale32 path unmapped. The
current documentation closes that exact live-path gap; all three checklist
items remain pending a fresh live-bound Reviewer decision.
`design/SPEC.md.orig` remains historical and non-normative. The Manager-owned
pipeline state remains `current_stage=stage1_software_diagnosis`; this
read-only-audit binding does not edit or reinterpret it as `rtl`. No simulation,
formal, synthesis, timing, PPA, FPGA, U280, delivery, or product-completion
result is claimed. The preserved certified PPA belongs to its historical RTL
hash and does not cover the current shell.

## Historical standalone dynamic Scale32 successor

This section is retained as superseded historical context. Dynamic Scale32 is
not the current publication mechanism or an active successor.

- Contract: `shared_token_group_dynamic_scale32_v1`.
- Architecture source: the exact Manager-frozen proposal at
  `evidence/shared_token_group_dynamic_scale32_v1/architecture/PROPOSAL.json`
  (SHA-256 `b55ab977574dc2bdeb760e8859ec2fa49f9f835846e3da24bd0f887d84a74f41`),
  Manager freeze SHA-256
  `175d1dcc7016df0c94fb6ec0d086d78fe6b0e4b80f26d1ef42f746825b53c2d8`,
  and independent environment decision SHA-256
  `910efb2fd11e53f4404905b7f6e8e0a528986b873bad15dfe8da5ec522907eff`.
- The Manager advanced `environment -> rtl` at
  `2026-08-02T07:08:41.858846Z` and authorized bounded RTL implementation only.
  Verification, shell regression, canonical PPA, prototype, benchmark, and
  signoff remain locked.
- The Manager reconciled only the duplicated `successor` projection in
  `research/PIPELINE_STATE.json`; its companion hash is the binding authority
  for the Engineer remediation packet. This reconciliation records the prior
  transition and completed handoff and is not a new stage transition. The
  reconciled state SHA-256 is
  `d41df90313bc683b7eab32a1615e3344554d3bf27a811228ebf2bbc7706ab31c`.

### Requirement-to-module mapping

| Frozen requirement | Synthesizable implementation | Focused evidence |
| --- | --- | --- |
| 64-lane Q/K/V-head groups and 128-lane hidden/MLP groups | `ace2_dynamic_scale32_group_core` accepts exactly 64 or 128 lanes | generated and bit-exact simulated 64/128-lane vectors plus per-top Icarus elaboration |
| One 1,280-byte ping-pong wide-group buffer reused across mutually exclusive producers | Two `128 x signed-40` banks in `ace2_dynamic_scale32_group_core` total exactly 1,280 bytes | Verilator lint and source inspection |
| Smallest legal delta in `[-24,+24]`, effective exponent in `[-24,+4]`, RNE, all-zero delta zero | Max-absolute selector scans legal deltas in ascending order; output quantizer uses signed ties-to-even shifts | Python scalar reference, five deterministic vector families, bit-exact RTL simulation |
| Signed-int8 symmetric `[-127,+127]`; `-128` reserved; no silent clipping | Payload is emitted only after a legal delta is found; no saturation path exists; impossible groups commit `numeric_overflow` without payload | RTL simulation checks every emitted lane and explicit fail-closed descriptor cases |
| Exact Scale32-tagged partial accumulation with no intermediate rounding | `ace2_scale32_tagged_accumulator_core` multiplies signed-32 partials by both normalized Scale32 significands, aligns to canonical exponent `-78`, and accumulates in signed-160 | scalar reference, exact two-event RTL vector, depth-4 Yosys SAT proof |
| One shared aligner/accumulator, at most 38 input groups | A single ready/valid accumulator accepts one event per cycle, bounds a transaction to `MAX_EVENTS=38`, and saturates its event counter after an excess-event descriptor error so it cannot wrap | lint, elaboration, 40-event fail-closed simulation, formal stall checks |
| Exact 64-byte little-endian `BFP1` sidecar | `ace2_dynamic_scale32_sidecar_builder_core` emits the frozen magic, schema, shape, producer identity, element count, model identity, deltas, and zero fill | byte-exact generated sidecar and RTL comparison |
| Validate magic/schema/shape/tag/layer/opcode/model identity/exponents/reserved bytes/address before payload issue | `ace2_dynamic_scale32_sidecar_validator_core` checks the complete header, payload alignment/non-wrap lower bound, group count, base Scale32 records, effective exponents, zero fill, and reserved tail | valid and corrupt-sidecar RTL cases plus scalar round trip |
| Independent ready/valid backpressure without changed payload, sidecar, tag, delta, or errors | Every result channel is registered and held until `ready`; group capture and emission use ordered alternating banks | multi-cycle group payload/commit, sidecar, and accumulator result stalls in simulation; formal accumulator result-stability proof |
| Single clock, asynchronous reset assertion, synchronous state release, no CDC | All four modules use only `clk_i` and active-low asynchronous reset; reset/clear invalidates bank, sidecar, accumulator, and result-valid state | lint, simulation X checks, bounded formal reset assertions |

### Icarus diagnostic-clean compatibility repair

- The five combinational processes formerly declared with `always_comb` are
  declared with the synthesizable Verilog-equivalent `always @*`. The group
  core exposes selected read/write-bank fields and emit lanes through
  continuous scalar/packed aliases so those processes do not directly read
  unpacked arrays. Interfaces, arithmetic, storage, state updates, and
  sensitivity semantics are unchanged; this avoids Icarus's nonfatal
  constant-select and unpacked-array sensitivity diagnostics without
  suppressing or filtering them.
- The integrated and four per-top elaboration logs must each be byte-empty.
  The recovery tool rejects any emitted byte and contains a synthetic
  diagnostic self-test that must exercise the rejection path.
- The minimal-formal extractor requires the live accumulator's new `always @*`
  declaration before applying only its existing Yosys parser shims; the proof
  harness, depth, and properties are unchanged.

### Interface and parameter freeze

- `ace2_dynamic_scale32_group_core`: `VALUE_WIDTH=40`,
  `LANES_PER_BEAT=16`, `MAX_GROUP_LANES=128`; exact wide values are expressed
  in immutable baseline-group units. Inputs are one start transaction followed
  by four or eight 16-lane beats. Outputs are provisional 16-lane signed-int8
  beats followed by one commit record carrying group index, tensor tag, delta,
  effective Scale32, and mutually exclusive error status.
- `ace2_dynamic_scale32_sidecar_builder_core`: one tensor header followed by
  exactly `group_count` delta/base-Scale32 pairs; a malformed shape or address
  is `descriptor_error`, while an illegal delta/effective exponent is
  `numeric_overflow`.
- `ace2_dynamic_scale32_sidecar_validator_core`: one 512-bit sidecar plus up to
  38 packed base Scale32 records and exact expected producer/tensor identity.
- `ace2_scale32_tagged_accumulator_core`: one start/tag, one to 38 signed-32
  partial events with two Scale32 records each, and one signed-160 result at
  canonical binary exponent `-78`.

### Hardware discipline and provenance

- The Planner-cycle incident set is preserved byte-for-byte under
  `evidence/shared_token_group_dynamic_scale32_v1/recovery/planner_cycle0_bypass/archive`.
  Engineer line-by-line dispositions and exact Planner-to-Engineer hashes are
  published in the adjacent recovery evidence. Unchanged source bytes are
  explicit adoptions, not retroactive acceptance of Planner execution or logs.
- All combinational outputs have defaults; every state element has one
  sequential driver; counters and arrays are explicitly bounded; there are no
  latches, delays, initial design state, real numbers, DPI calls, or other
  simulation-only constructs in the synthesizable source.
- Error outputs are mutually exclusive with descriptor error taking precedence.
  Result and payload fields remain stable under backpressure.
- RTL, reference, generator, vectors, testbench, formal harness, and build
  configuration are first-party operator-directed project-internal work. The
  exact non-redistribution provenance boundary is recorded in
  `design/DYNAMIC_SCALE32_IP_PROVENANCE.json`; no third-party IP is introduced.
  Exact hashes and deterministic regeneration commands are recorded in
  `design/RTL_MANIFEST.json` and
  `evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/PRECHECK.json`.
- No third-party or generated synthesizable IP is introduced.

The new Dynamic Scale32 cores remain standalone and are not admitted to
`ace2_shell`; candidate capability acceptance remains false. Independently of
that candidate state, the authoritative ordered frontier is now certified
exactly through `layer_0.rope_q`, with `layer_0.rope_k` first unsupported and
mode `ADVANCE`. This publication is bound to RTL tree SHA-256
`e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be`
and canonical SKY130 packet aggregate SHA-256
`03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4`:
62,408 cells, 0.6126137952 mm² non-SRAM area, +0.3565 ns worst setup
slack, and 0.00 ns WNS/TNS at 100 MHz. The accepted certification decision is
`CERTIFY_ADVANCE_THROUGH_LAYER_0_ROPE_Q`, with Fresh Reviewer status `done` in
`handoff:rtl-rope-q-current-tree-frontier-certification-v1/round-0001.json`.
No execution flow was run for this metadata publication, and no certification
of `layer_0.rope_k` or later operator, baseline/model run, physical design,
prototype, benchmark, signoff, tapeout-readiness, or silicon result is claimed.

## RoPE-K frontier publication

The authoritative ordered frontier is now published exactly through
`layer_0.rope_k`; `layer_0.kv_write` is the first unsupported operator and no
KV-write result is claimed. The publication is bound to current RTL tree
SHA-256 `e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be`
and the Fresh Reviewer-accepted bounded replay at
`handoff:rtl-layer0-minimal-reference-capture-batch-replay-v1/round-0001.json`
(SHA-256
`06735fb358ea78a6b050de1adf57c904e0d4014f01cbc1a16367b37b19c64507`).
The accepted replay matched all 64 RoPE-K outputs exactly, reported zero
mismatches and zero reference-saturated elements, completed every ready/valid
exchange without timeout, and ran for 68,021 simulation cycles.

The one-shot capture witness is
`evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2/witness.json`
(SHA-256
`835c9d8f4432b2005998bac74c5536a9acd171f8042c33ac537af37d18869c3d`),
with capture `SHA256SUMS` SHA-256
`7e3c768d9eff34b327859621e0e093243cf7562837611ad0bd37480a86703e97`.
The ordered-prefix report SHA-256 is
`0ca5178f7d186640769643e0fa25aad312a3097afacc855cebd03a7c87aa5e21`,
the replay log SHA-256 is
`55f0671e71b91e081a30a227b2579bc3d8c3ab7824d8d96553c4be65f2f6aed6`,
and replay `SHA256SUMS` SHA-256 is
`ceade58ef0e5d10cdf538b4593712bd3688871757f39d3e36afe8ecb2d5704ac`.
The precise `layer_0.kv_write` boundary is retained in `boundary_probe.json`
at SHA-256
`9abaea3f004f8c141abbb0ccadbe75d0dec6c09f72a26e54f21f721367dbd49c`:
the maintained interface requires both KV heads and joint metadata, while the
accepted capture contains only head 0.

Canonical SKY130 PPA was not rerun. The unchanged current-tree packet remains
PASS at 62,408 cells, 0.6126137952 mm² non-SRAM area, 100 MHz, +0.3565 ns
worst setup slack, and 0.00 ns WNS/TNS. Its aggregate `SHA256SUMS` SHA-256 is
`03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4`.
No RTL, test, evidence, PPA, model, baseline, candidate, benchmark, Scale32,
pipeline, authority, or seal file was changed by this publication. Fresh
Reviewer acceptance of this finalized record set remains pending; this section
does not claim that acceptance.

## KV-write frontier publication

The authoritative ordered frontier is now published exactly through
`layer_0.kv_write`; `layer_0.attention_score` is the first unsupported operator
and no score result is claimed. The publication is bound to current RTL tree
SHA-256 `e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be`
and accepted mission `rtl-kv-write-two-head-reference-batch-replay-v1`. Fresh
Reviewer acceptance is recorded in
`handoff:rtl-kv-write-two-head-reference-batch-replay-v1/round-0001.json`
(SHA-256
`7423019ace4d58a1b1b376afc13b1e4247c02d65dbaf7071db5cf367a7184046`).

The immutable head-0, head-1, and logical merged packages are bound by their
respective `SHA256SUMS` SHA-256 values
`7e3c768d9eff34b327859621e0e093243cf7562837611ad0bd37480a86703e97`,
`6c7d6bcbf8e5034ffd88df0eb074da83959029de4b32ba0681fb310f9fe49228`,
and `4b6926066e9a8ca97e44d759653ff12a3c2ae642ca5ed4ecad11c8b0fee7ddd0`.
The head-0 witness SHA-256 is
`835c9d8f4432b2005998bac74c5536a9acd171f8042c33ac537af37d18869c3d`,
the head-1 witness SHA-256 is
`a13b1cf391bff1609c58e8c246e4c2cd3b8ddc21013b23fc3889a57049dbd9a2`,
and the logical merge manifest SHA-256 is
`b805226718f49bed2d2168d259596221c69ab6d1dc7317336388aa85480102c7`.

Independent unchanged-shell replay matched all 17 cache writes exactly in 188
command cycles: 128 K bytes, 128 V bytes, and 16 metadata bytes, with no
dropped or misordered beat and no payload mismatch. The ordered-prefix report
SHA-256 is
`5be60b3d287be14e4c5c6a5d7cb1f52454be32799c6616b8c8f903e21a73c2de`,
the replay log SHA-256 is
`3a461863c9e6d97b480617f296dcbb75eefe7d9fc487f4f942bf0a6ec8458552`,
and replay `SHA256SUMS` SHA-256 is
`ee23b5846fc17f479add74fa6a05646846421e0f05b3608455bf109648b00985`.
The boundary probe SHA-256 is
`ace66c3020d109f14d9a35cbc4360c6e3726f142d0695d9f2ad1b1a0dbc0ba92`:
the retained coordinate lacks the RoPE-Q query and token-0 K payload needed by
`layer_0.attention_score`, and no later operator was attempted.

Canonical SKY130 PPA was not rerun. The unchanged current-tree packet remains
PASS at 62,408 cells, 0.6126137952 mm² non-SRAM area, 100 MHz, +0.3565 ns
worst setup slack, and 0.00 ns WNS/TNS. Its aggregate `SHA256SUMS` SHA-256 is
`03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4`.
No RTL, test, evidence, PPA, model, baseline, candidate, benchmark, Scale32,
pipeline, authority, or seal file was changed by this publication. Fresh
Reviewer acceptance of this finalized record set remains pending; this section
does not claim that acceptance.

## Attention-score and softmax frontier publication

The authoritative ordered frontier is now published exactly through
`layer_0.softmax`; `layer_0.attention_value` is the first unsupported operator
and no attention-value result is claimed. The publication is bound to current
RTL tree SHA-256
`e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be`
and accepted mission `rtl-attention-score-token0-reference-batch-replay-v1`.
Fresh Reviewer acceptance of that bounded replay is recorded in
`handoff:rtl-attention-score-token0-reference-batch-replay-v1/round-0001.json`
(SHA-256
`43fbfdb9b627ec970972294211c01e2ec587a74759d06141107b513330418f15`).

The immutable one-forward token-0/head-0 K capture is bound by witness SHA-256
`8aada02757f3886a83dbc2219a56b9576a3d6154728be4f40ea7e321e217a2fe`
and capture `SHA256SUMS` SHA-256
`801d777b09a0d14500fca14bf8e03ae4e9bb5eede9b2a034021b91c671f2bdb3`.
It retained only the 64-byte token-0 K projection and 64-byte token-0 RoPE-K
cache payload; their SHA-256 values are respectively
`808b0ac5094ec4781c43667791895d3650eb9055d00995b95ad6973c674b9037`
and
`61eee1f2b5278415d544291239aefa7961fff4f15100c8bbcd436b9d09922ab3`.
The hash-compatible, source-package-preserving attention-input merge manifest
SHA-256 is
`8b9d0d51e7394e3e38df5a808d0e0e95fab6fcf519f28383e7ad050c1b18444d`.

Independent unchanged-shell replay matched both active-token attention scores
exactly with zero mismatches in 1,871 command cycles. It then matched both
active-token softmax probabilities exactly with zero mismatches, Q0.15 sum
32,768, and 147 command cycles. The ordered-prefix report SHA-256 is
`1c7fda411e29449b1596e38e0d1cf44f94b38c220237e1e16db48cb26f38d115`,
the replay log SHA-256 is
`3a8e04ba8b2e4e0bb88352357e8d35e87406da6bdc2fe5d1fa4a5b5192b545e6`,
and replay `SHA256SUMS` SHA-256 is
`22a3addf99fd8afe757e62c303add0dbc5a97726a675550d8870aafcaf298604`.
The precise `layer_0.attention_value` boundary is retained in
`boundary_probe.json` at SHA-256
`e39ab52fb9dd215d050919b0d07f261a6075d6cfec3caf26e8344e5a0e9c612f`:
token-0/head-0 V is not retained, so the exact executable interface is
unavailable and no later operator was attempted.

Canonical SKY130 PPA was not rerun. The unchanged current-tree packet remains
PASS at 62,408 cells, 0.6126137952 mm² non-SRAM area, 100 MHz, +0.3565 ns
worst setup slack, and 0.00 ns WNS/TNS. Its aggregate `SHA256SUMS` SHA-256 is
`03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4`.
No RTL, test, evidence, PPA, model, baseline, candidate, benchmark, Scale32,
pipeline, authority, or seal file was changed by this publication. Fresh
Reviewer acceptance of this finalized record set remains pending; this section
does not claim that acceptance.

## Attention-value frontier publication

The authoritative ordered frontier is now published exactly through
`layer_0.attention_value`; `layer_0.o_proj` is the first unsupported operator
and no output-projection result is claimed. The publication is bound to current
RTL tree SHA-256
`e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be`
and accepted mission `rtl-attention-value-offline-v-authority-replay-v1`.
Fresh Reviewer acceptance of that bounded mission is recorded in
`handoff:rtl-attention-value-offline-v-authority-replay-v1/round-0002.json`
(SHA-256
`47c946da8733a1a2cc99470fba188e00ed6700458ce28c636053797bb72ac9f3`).

The forward-free token0/head0 V authority payload SHA-256 is
`ae7b9d41e1b6fe625b111dc2b118f5007c821df23c036d927c25ca9e40fca565`.
It was hash-bound with accepted softmax SHA-256
`fdda20da03213e1cb0c63b7575d4397948b55f5bda4d096310266972b5d6fa54`
and immutable token1/head0 V SHA-256
`341181d2a9c28549cac85a6c9b4532283b7e29d82e4011d8bef92f826f19a247`.
The canonical component-record SHA-256 is
`55dcfcbc54d392d62359fe349a4bd2dcb05d6450b7c257d2064f5f54a7103c2e`
and the ordered 132-byte input payload SHA-256 is
`898474e64f8de2ee5f9ce1573d2e0fa17918f47258951b4696f7220805026f48`.
The source-package-preserving merge manifest SHA-256 is
`95d6a51a52fb9f2b49d63b55da853ec528715d0df5f42d48b7ba5ed8763e860d`;
its `SHA256SUMS` file hashes to
`24d9b9054bd34fe59513fdb529e1796333d9422cbc3e1f0177d3da5108ce71f5`.

Independent unchanged-shell replay matched all 64 attention-value outputs
exactly with zero mismatches and zero saturation in 1,817 command cycles. It
emitted four RTL writes, retained Q0.15 probability sum 32,768, and produced
output SHA-256
`a4bed51f31c0179ed0ba91db3bb75ef2435037041bb6bf773111598b8a5c8c48`.
The ordered-prefix report SHA-256 is
`d03d8171dee9e2a7af6be06f70b7c97915a022c8aeb2ba115f17bfc363e57717`,
the replay log SHA-256 is
`e5d9ae7edcd5e2588ea1920d03ca91b57044f60493c03b90e0f632fc0c7806c4`,
and replay `SHA256SUMS` SHA-256 is
`ea258971aebee28c7ea680e9f4272c37b08c55afcdb77494011827107af8f199`.
The precise `layer_0.o_proj` boundary is retained in `boundary_probe.json` at
SHA-256
`eb410d4598a4c9c6bd6915c0f2e1e35083a86498940377a56fe602e723d68967`:
the exact 896x896 signed-int4 weights and 896-channel bias, multiplier, and
right-shift metadata are unavailable, so no output-projection or later
operator was attempted.

Canonical SKY130 PPA was not rerun. The unchanged current-tree packet remains
PASS at 62,408 cells, 0.6126137952 mm² non-SRAM area, 100 MHz, +0.3565 ns
worst setup slack, and 0.00 ns WNS/TNS. Its aggregate `SHA256SUMS` SHA-256 is
`03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4`.
No RTL, test, evidence, PPA, model, baseline, candidate, benchmark, Scale32,
pipeline, authority, or seal file was changed by this publication. Fresh
Reviewer acceptance of this finalized record set remains pending; this section
does not claim that acceptance.

## Output-projection frontier publication

The authoritative ordered frontier is now published exactly through
`layer_0.o_proj`; `layer_0.attention_residual_add` is the first unsupported
operator and no residual-add or later result is claimed. The publication is
bound to unchanged RTL tree SHA-256
`e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be`
and accepted mission `rtl-o-proj-offline-metadata-batch-replay-v1`. Fresh
Reviewer acceptance of that bounded mission is recorded in
`handoff:rtl-o-proj-offline-metadata-batch-replay-v1/round-0001.json`
(SHA-256
`25f2ed3663b1351b4d5a92b0b23b0ed964ba360a1453eb541fa2cdbfc115aa94`).

The zero-model authority binds Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, raw tensor
`model.layers.0.self_attn.o_proj.weight` with shape 896x896, raw BF16 tensor
SHA-256
`8450c5e779c06840601c0b7420769b78cd2a06293cba8fe0ea172e40411dcdab`,
and safetensors SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
The derived 802,816 signed-int4 weights hash to
`17cd84c9ac31fe6ff478f8114ab873a62446c6fb53a7bc33e6bb9af8d772f6c2`.
The 896-channel multiplier and right-shift arrays hash respectively to
`cbef63fc80a5204c787789bfff54e7b3fc321e460b3fb297f1e873fcd893f231`
and
`be9a9f52ee08d2d4eb1205bf1ffcb19fbab0bb6807ec03ede0ef389f1cd809d2`;
the 896 zero bias accumulators hash to
`6cf1b57d59e7111bc218dfb01dda93ac0f776715599a1c69f89035bd20c16a10`.
The complete derived-metadata record SHA-256 is
`7a21592c32c396febfec76e5451fe3fc38cf9f563a1d925d8d7b28387aa5954d`.

Independent unchanged-shell replay matched all 896 output-projection outputs
exactly with zero mismatches and zero saturation in 2,566,373 command cycles.
It emitted 56 ordered RTL writes, passed the maintained watchdog and
dropped-write checks, and produced output SHA-256
`5217926db46bd1ffdbc9d9ad51ec11e1fc8295064041fd12fe011049c62bafb6`.
The ordered-prefix report SHA-256 is
`69fc76ae1c6663c5ca8e0ab963e1d354e30fcc031fa791a01aad3d4fa089cef6`,
the replay log SHA-256 is
`5ccfc19fdad8e8268a3684a596b64e4e2a9762916b3642faaebaa63f6ddfbeb4`,
and replay `SHA256SUMS` SHA-256 is
`0f3096299647b33e7dcb841a0429d4002811f6da2d7d4e06c5887db1fb248a10`.
Zero model, forward, call, calibration, hook, capture, or extraction execution
is recorded by `zero_model_execution.json` at SHA-256
`97c99a9856bcae66b59a8e1a0c230b21d541fc47e806a8cdba432f75cbdeeef7`.

The precise `layer_0.attention_residual_add` boundary is retained in
`boundary_probe.json` at SHA-256
`4061fb3f8d44d66749ec7398f1492795bc3e2469a4910f46e02300017a87f5ca`:
the independently retained residual-source operand at the frozen destination
scale and its hash-bound two-operand descriptor payload are absent, so no
residual-add or later operator was attempted.

Canonical SKY130 PPA was not rerun. The unchanged current-tree packet remains
PASS at 62,408 cells, 0.6126137952 mm² non-SRAM area, 100 MHz, +0.3565 ns
worst setup slack, and 0.00 ns WNS/TNS. Its aggregate `SHA256SUMS` SHA-256 is
`03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4`.
No RTL, test, evidence, PPA, model, capture, baseline, candidate, benchmark,
Scale32, pipeline, authority, or seal file was changed by this publication.
Fresh Reviewer acceptance of this finalized record set remains pending; this
section does not claim that acceptance.

## Complete Layer-0 frontier publication

This section is retained as append-only historical publication state and is
superseded by the full-Qwen final-tree publication below.

The authoritative ordered frontier is now recorded as the complete 18-operator
Layer-0 prefix through `layer_0.mlp_residual_add`. There is no unsupported
Layer-0 operator. The next product boundary is one coherent full-Qwen
multi-layer/autoregressive integration mission covering shared-layer reuse,
cross-token KV-cache sequencing, embedding, final norm, LM head, and the
autoregressive generation loop; no result at that boundary is claimed here.

This publication binds unchanged RTL tree SHA-256
`e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be`
and authority-bundle `SHA256SUMS` SHA-256
`11cb9f5b1a1947091cb3bcace6111539bc1b7cf1e2defe5f12422dcc1077b7be`.
The accepted ordered-prefix report SHA-256 is
`cbcaa2118f13bf252c935809197162a597be5a967213ed6f479e183f8bfa2014`;
its batch result is `LAYER0_COMPLETE_EXACT` and its first failing or
unsupported operator is null. Fresh Reviewer acceptance of the bounded replay
is recorded in
`handoff:rtl-layer0-remaining-offline-authority-batch-replay-v1/round-0001.json`
at SHA-256
`1d788099bc200dc54d071a8e4863fa314c06a9b3ae98de399d1209067db0f4fa`.

All seven newly certified results are exact with zero mismatches:
`attention_residual_add` 896/896, `post_attention_rmsnorm` 896/896,
`mlp_gate_proj` 4864/4864, `mlp_up_proj` 4864/4864, `silu_gate` 4864/4864,
`mlp_down_proj` 896/896, and `mlp_residual_add` 896/896. The authority records
`model_execution_count=0` and `model_subprocess_count=0`; its
`zero_model_execution.json` hashes to
`db19e3046038e417d54927d91fcb612eca8544b10c05f96ac6daaf16d62d74d0`.

Canonical SKY130 PPA was not rerun. The unchanged current-tree packet remains
PASS at 62,408 cells, 0.6126137952 mm² non-SRAM area, 100 MHz, +0.3565 ns
worst setup slack, and 0.00 ns WNS/TNS. Its aggregate `SHA256SUMS` SHA-256 is
`03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4`.
No RTL, test, evidence, authority, seal, model, capture, replay, synthesis,
OpenSTA, PPA, benchmark, baseline, candidate, or Scale32 flow was changed or
run by this publication. Fresh Reviewer acceptance of this finalized record
set remains pending; this section does not claim that acceptance.

## Full-Qwen final-tree functional acceptance and canonical timing NO_GO publication

The authoritative current RTL binding is the accepted 23-file final tree at
SHA-256
`35da3c7ca116eb3031b9d86cdbdda3b6faab5369d694f438aff116c08dad1947`.
Layer 0 remains exact for all 18 of 18 ordered operators with no unsupported
Layer-0 operator. The accepted full runtime completes all 13,914 of 13,914
commands across 24 layers and two token steps, produces token IDs `[0, 0]`,
records `first_failure=null`, and consumes 1,240,410,384 simulator cycles.
That functional integration is accepted; it is not a timing-closure or product
sign-off claim.

The Layer-0 report is hash-bound to the earlier accepted RTL tree
`e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be`.
Its embedded `PENDING_FRESH_REVIEWER` field is resolved by the sealed Reviewer
decision at
`handoff:rtl-layer0-remaining-offline-authority-batch-replay-v1/round-0001.json`
(SHA-256
`1d788099bc200dc54d071a8e4863fa314c06a9b3ae98de399d1209067db0f4fa`,
status `done`). Layer-0 acceptance is carried to the final tree by exactly two
Fresh-Reviewer-accepted shell-only deltas: `e10d78dd...0f8be` to
`3312e4f6...b46b` via
`handoff:rtl-silu-packed-int8-width-adapter-repair-v1/round-0002.json`
(SHA-256
`43a165c835a785ffb9db19eb5592785dbc284f72aa4ef06f4caa6c9ffe6ecb3f`),
then `3312e4f6...b46b` to `35da3c7c...1947` via
`handoff:repair-full-qwen-final-lm-head-tile-boundary-v1/round-0001.json`
(SHA-256
`907bb8de62d980ee0e367ac55a761877fef9669f95469570f976d87ae9c158c1`).
Both deltas change only `rtl/ace2_shell.sv`; the Layer-0 arithmetic RTL remains
the independently accepted implementation.

The runtime binds schedule SHA-256
`838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002`,
sealed image SHA-256
`e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb`,
and raw-model SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
The accepted runtime validation is
`evidence/verification/repair-full-qwen-final-lm-head-tile-boundary-v1/runtime_validation.json`
at SHA-256
`ac09ee13ef6b25491510c2f4f947486eabeface10c63c1e15e65bfd53806e212`.
Fresh Reviewer acceptance of that runtime repair is bound by
`handoff:repair-full-qwen-final-lm-head-tile-boundary-v1/round-0001.json`
at SHA-256
`907bb8de62d980ee0e367ac55a761877fef9669f95469570f976d87ae9c158c1`.

The sole canonical final-tree SKY130 evidence is
`evidence/canonical_sky130/rtl-full-qwen-final-tree-canonical-sky130-ppa-v1/`.
Its independently accepted metrics are 62,222 mapped cells and
0.6137373728 mm² non-SRAM area, which passes the 2.0 mm² cap, but the final
tree misses the 100 MHz floor by 0.1484 ns. Worst setup slack is -0.1484 ns,
WNS is -0.15 ns, and TNS is -6.28 ns. The critical `clk_i` max path starts at
`_44008_`, ends at `_46186_`, arrives at 9.8432 ns, and is required at
9.6948 ns. Therefore the accepted canonical gate verdict is `NO_GO`, and
final product timing is not certified.

The independent metrics file hashes to
`4161b6b80dd9b10db11cad56eedb32324638b28fd53986a28141730fbca579cb`;
the aggregate `SHA256SUMS` hashes to
`4336c03c2c1db9fc761c31b37650986a481cebd1d57dd349b6d6359032a96ea5`.
Fresh Reviewer acceptance of the exactly-once PPA packet is bound by
`handoff:rtl-full-qwen-final-tree-canonical-sky130-ppa-v1/round-0001.json`
at SHA-256
`98df9f192c4d8d73065f7aac3cd300558a39e32b0188e5e2ce7e577094240241`.

This publication ran no PPA, simulation, runtime, model, image, benchmark,
synthesis, or STA flow and changed no RTL, schedule, image, model, runtime
evidence, or PPA evidence. It makes no deployable, routed, power, sign-off,
GDS, pre-tapeout-readiness, tapeout, or silicon claim. Its own cross-file
publication review remains pending. Exactly one successor is queued, blocked
on that Fresh Reviewer acceptance: diagnose and repair only the accepted
`_44008_ -> _46186_` cone while preserving the accepted runtime behavior.
Canonical PPA remains prohibited for the repaired tree until that tree is
Fresh-Reviewer accepted and a separate PPA authorization is issued.

## Critical-cone-repaired tree canonical timing NO_GO publication

This append-only publication supersedes the current pointers of the preceding
final-tree timing record without deleting or rewriting that history. The
accepted 23-file RTL tree is now SHA-256
`e623db3663ef3854616ce365c51aaef36e8915e0a60a7bb936c4d6c34c0b8fa0`;
its repaired `rtl/ace2_shell.sv` is SHA-256
`8bf63ef4bb98700e2f3b86ae4faf1883da76d3875e8632f086453cc51d15f091`.
The source-to-netlist repair is
`rtl-final-tree-critical-cone-timing-repair-44008-to-46186-v1`, bound by
`evidence/verification/rtl-final-tree-critical-cone-timing-repair-44008-to-46186-v1/repair_witness.json`
(SHA-256
`b8964b76bd65606bd6465a8e78924bbf262b0ebdf718a249257d48b0e7bd1f54`),
its aggregate `SHA256SUMS` (SHA-256
`42c288e7299db1ff2d905dfa345cd460e70cd571f208724ec9728bf93bdde4d5`),
and Fresh Reviewer decision
`handoff:rtl-final-tree-critical-cone-timing-repair-44008-to-46186-v1/round-0001.json`
(SHA-256
`bfc419be270a44922d9e2f9bd9cc4a50b992f19a6cf99bade4ffb4c197dc1db6`).
Its focused regression passed all 74 runs in 47,085,087 total cycles.

Functional authority remains accepted and unchanged: Layer 0 passes 18 of 18
ordered operators, and the full-Qwen two-token runtime passes all 13,914 of
13,914 commands with generated tokens `[0, 0]`. The schedule remains SHA-256
`838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002`
and the sealed image remains SHA-256
`e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb`.
This accepted functional evidence is not a timing-closure or sign-off claim.

The sole canonical repaired-tree SKY130 evidence is
`evidence/canonical_sky130/rtl-critical-cone-repaired-tree-canonical-sky130-ppa-v1/`.
It reports 62,327 mapped cells and 0.6138362176 mm² non-SRAM area, so the
2.0 mm² cap passes. Detailed setup slack is -0.5275 ns, WNS is -0.53 ns,
and TNS is -67.68 ns at the unchanged 10.000 ns clock. The critical `clk_i`
max path is `u_rmsnorm_core/_7483_ -> u_rmsnorm_core/_7667_`, terminating at
the capture cell's `DE` pin. The explicit 100 MHz verdict is `NO_GO`, and
final product timing is not certified.

Relative to the preceding final tree, the repaired tree adds 105 cells and
0.0000988448 mm², while detailed setup slack changes by -0.3791 ns and TNS
by -61.40 ns. The shell repair removed the prior `_44008_ -> _46186_`
control cone; it did not improve global timing and exposed or worsened a
structurally different RMSNorm enable/capture cone.

The canonical `FINAL_METRICS.json` and `FINAL_INDEPENDENT_METRICS.json` hash
respectively to
`535398bdae3fb30c83db38ae033c785b5d09c48fefe1a35b19b9b6520184647a`
and
`0d5266df7452283fe5a4df3efc6fc954ba12b17f5a013552c0cf4d5157d41a8a`;
the aggregate `SHA256SUMS` hashes to
`21c75b187ea13d8624f4d0922ea7ad9a30b720298e36045072a692fc3f0cc479`.
Fresh Reviewer acceptance of that exactly-once PPA packet is bound by
`handoff:rtl-critical-cone-repaired-tree-canonical-sky130-ppa-v1/round-0001.json`
at SHA-256
`8352eeefaf77da0946612ff6b48519fdde7b4412a967f47e0fe53e36f8018997`.

This publication ran no PPA, synthesis, STA, simulation, runtime, model,
image, or benchmark flow and changed no RTL or engineering evidence. Its own
cross-file publication review remains pending. Exactly one already
materialized successor,
`rtl-rmsnorm-critical-cone-repair-7483-to-7667-v1`, remains dependency-gated
on that acceptance; no duplicate was created, and canonical PPA is forbidden
inside the repair task.

### Complete live-tree file provenance

The repaired-tree manifest now gives every live synthesizable source a
path-qualified provenance record. In particular,
`rtl/ace2_cross_layer_error_carry_core.sv` is bound at 43,912 bytes and
SHA-256
`6a8286e95b77cb7b9e4c3d476677f6eac60ae12039a1431cf981fc4deba80e8d`
to its three declared modules and frozen
`cross_layer_quantization_error_carry_final_output_v1` contract.
`rtl/ace2_tile_max_delta_attention_core.sv` is bound at 17,584 bytes and
SHA-256
`be7f1a2290a47d9d5dba77aff3e11babcfd546e5b99304c65d46987e1bf76c36`
to `ace2_tile_max_delta_score_core` and
`ace2_hierarchical_softmax_core` under the
`layer0_tile_max_delta_attention_v1` contract. These records add metadata
traceability only; neither source file nor any engineering evidence changed.

## RMSNorm-repaired tree canonical timing NO_GO publication

This append-only publication supersedes current pointers for publication
`frontier-publication-2026-08-04T04:43:19Z` without deleting or rewriting it. The accepted
23-file RTL tree is SHA-256 `a42bc8469b929a1cc193fefb45ff630457db9d298b0f611571992a79bacb360a` and `rtl/ace2_rmsnorm_core.sv` is SHA-256
`18177d9fc5eefaf550e129fd13aa908d16f2b2d24a8925ed97e81cf4cf38d5b0`. The accepted bounded repair is `rtl-rmsnorm-critical-cone-repair-7483-to-7667-v1`, bound by its
repair witness (SHA-256 `314428e242b6b6382f66d05ac8a781661bdd91b6cf7e28716414f0ffd8657300`), aggregate
`SHA256SUMS` (SHA-256 `e6624837c95c0a4c29563e787158c1f08fe3d30d7eebf76a5c03f2d5ee646919`), and Fresh
Reviewer decision (SHA-256 `1f51ea77ef54d061042e316ebd03ee4c688aa751f04327bf528b83d3bcbb6bdb`).

Functional authority is preserved: Layer 0 passes 18 of 18 ordered operators,
and the full-Qwen two-token runtime passes 13,914 of 13,914 commands with tokens
`[0, 0]`. The accepted RMSNorm repair is semantics-preserving and changed only
`rtl/ace2_rmsnorm_core.sv`; this publication performed no functional replay.

The sole canonical PPA packet is `evidence/canonical_sky130/rtl-rmsnorm-repaired-tree-canonical-sky130-ppa-v1/`. It reports 62,330 cells,
0.61393256 mm2 non-SRAM area PASS, -0.1741 ns detailed setup slack, WNS
-0.17 ns, and TNS -0.29 ns. The explicit 100 MHz verdict is `NO_GO`,
so final product timing is not certified. Relative to the e623 tree, the deltas
are +3 cells, +0.0000963424 mm2, +0.3534 ns detailed slack, +0.36 ns WNS, and
+67.39 ns TNS.

The critical path is `u_rmsnorm_core/_7721_ -> _7482_`, source-mapped as
`collect_square_q[0] -> div_dividend_q[47]`. The causal final logic is the
`next_sumsq_w + HIDDEN_HALF_ACC` carry cone. The canonical `RESULT.json`,
`METRICS.json`, `INDEPENDENT_METRICS.json`, and aggregate `SHA256SUMS` hash to
`547a51704342f0f28348b460c8ddcd63153876d299d421826d5b176e828cf5ca`, `61a718f6d5598fcea332a2a668bf318ab39c73bc0752ec7b7353469c1588d1e4`,
`a6d6f81b8dd83d07ae3945dc8daec161117ff2fec997082d9ef3bf797b17af45`, and `b683c1d070d53aaf73edd23778a92c30a3bf19e643e4e9fd98bfffd3d51dcb85`.
Fresh Reviewer acceptance of the exactly-once PPA packet hashes to
`06d2938dbd5109acd356c7ce0ab4ff7e7f537705ab3ba25575814f0320ec0e8d`.

This publication ran no RTL, simulation, formal, synthesis, STA, PPA, runtime,
model, image, benchmark, prototype, or signoff flow and changed no engineering
evidence. Its own cross-file publication review remains pending. Exactly one
dependency-gated successor, `rtl-rmsnorm-final-sumsq-carry-cone-repair-v1`, is materialized; it may be released
only after Fresh Reviewer acceptance of this publication. PPA/Yosys/OpenSTA/
OpenROAD and broad functional replay are forbidden inside that repair.

## Final product publication for the demonstrated mapped-SKY130 scope

Publication `final-product-publication-2026-08-04T07:26:46Z` supersedes the current pointers of
`frontier-publication-2026-08-04T06:00:28Z` while preserving that NO_GO record in append-only
history. The accepted final 23-file RTL tree is SHA-256 `bf12e2c83b4d569b27bbbc14835ed8d36c39ec4e8820725cc7ad054fd7ffb4f6`. Its final
bounded repair is `rtl-rmsnorm-final-sumsq-carry-cone-repair-v1`, bound by
`evidence/verification/rtl-rmsnorm-final-sumsq-carry-cone-repair-v1/rtl_tree_manifest.sha256` (SHA-256 `bf12e2c83b4d569b27bbbc14835ed8d36c39ec4e8820725cc7ad054fd7ffb4f6`), aggregate
`SHA256SUMS` (SHA-256 `7f4e4ac2abfc9fdb3e9410d59316a108835919dd513786bd9a8cd4e788f2d688`), and Fresh
Reviewer handoff `handoff:rtl-rmsnorm-final-sumsq-carry-cone-repair-v1/round-0002.json`
(SHA-256 `871ad6f913cc6e14331472c71efcb9b9a985da6a495f251f92e28f01b5194f42`).

Functional authority is Layer-0 18/18 exact PASS plus full-Qwen 13,914/13,914
commands across 24 layers and two token steps, generated tokens `[0, 0]`, no
first failure, and 1,240,410,384 cycles. The schedule, image, and raw-model
SHA-256 values are `838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002`, `e24e0365e9fad5df2efe3e40df12e3f89f951f37c83449cb40e7d18fb614eafb`, and `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`; the raw model
revision is `060db6499f32faf8b98477b0a26969ef7d8b9987`.

The exactly-once canonical packet is `evidence/canonical_sky130/rtl-final-sumsq-repaired-tree-canonical-sky130-ppa-v1/`. It reports 62,283 cells,
0.614082704 mm2 non-SRAM area, +0.6966 ns detailed setup slack, WNS 0.00 ns,
TNS 0.00 ns, and PASS at 10.000 ns / 100 MHz in SKY130 HD TT 25C 1.80V.
`RESULT.json`, `METRICS.json`, `INDEPENDENT_METRICS.json`, `SHA256SUMS`, and the
authorization-consumption marker hash to `06d7bcb241885e18a97a9acf9a3af980441d4ac64a6b58655dee726f799a1042`,
`b5d8ae3806c5504d9dc4f294734be705016a2c4a8d14d13f3f6c167debfd9916`, `fe81bfc3c36a76e88b40d4056448ce98f607c1bacb7cdd078df813e6a3b98985`,
`7cbf50a2bcd2ee01d9898ebbe75f14b4620a66c4e24a098b55163791e3c2f47c`, and `75b8735c46a337283e3c4e0595c909434e7550aa69a7265d0a48b5888646d36c`.
Fresh Reviewer PPA acceptance hashes to `05576278ede0745edbcbd5a8e00e1f828fdc461a9c486c9045433ed225e9f148`.

The product is complete only for functionally integrated 24-layer/two-token
Qwen command execution plus mapped SKY130 synthesis/OpenSTA at 100 MHz and
within 2.0 mm2 non-SRAM. This does not claim routed timing, power signoff,
DRC/LVS, GDS or tapeout, silicon validation, longer-token generation, FPGA, or
external deployment interfaces. This publication ran no engineering flow,
changed no engineering evidence, enqueued no successor, and issued
`research/FINAL_PRODUCT_CERTIFICATE.json` pending independent Fresh Reviewer
verdict `FORMAL_PRODUCT_CERTIFIED`.

## Publication-integrity ordering correction

The pending publication was metadata-corrected at `2026-08-04T07:43:34Z`.
`make publication-authorization-preflight` now runs the companion/integrity
checker and its ordering regression before any future exactly-once runner may
create an authorization marker. The checker hash-locks the five already
consumed historical runners and requires every non-historical runner to place
that guard before marker creation. No RTL, constraints, flow output, canonical
PPA packet, verification evidence, or other engineering evidence was changed.

## Fresh Reviewer formal-product certification closure

At `2026-08-04T07:55:21.118678Z`, the exact round-0002 Fresh Reviewer verdict for
`manager-final-formal-product-publication-certification-v1` (SHA-256
`4e837bf2a8de724f9908919399a6637d41c653efdc669081badc43536a106941`) closed publication `final-product-publication-2026-08-04T07:26:46Z` with status `done` and
decision `FORMAL_PRODUCT_CERTIFIED`. The certificate remains limited to the
24-layer/two-token Qwen2.5-0.5B W4A8 integration and mapped SKY130
synthesis/OpenSTA scope already recorded above. No RTL, constraints, engineering
evidence, canonical runner, successor, routed-timing, power, DRC/LVS, GDS,
tapeout, FPGA, deployment, or silicon claim was added or changed.
