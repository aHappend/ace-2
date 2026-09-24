# ACE-2 productization specification and preserved two-token RTL contract

## Productization specification-stage addendum

This addendum is the active specification-stage contract for the ordered
ACE-2 productization objective. The older two-token 24-layer material below is
preserved as bounded predecessor context only; it is not a substitute for the
required arbitrary-text product demo, W4A8/RTL agreement, or U280 deployment
status. Live Planner-cycle-8 inspection found the Manager-owned
`research/PIPELINE_STATE.json` at `current_stage=stage1_software_diagnosis`
with `stages.specification.status=done_advanced_to_bounded_stage1_software_diagnosis`;
this document remains the frozen behavior, interface, protocol, acceptance, and
benchmark-closure contract. It does not authorize official model/evaluator
execution, `attempt-0003`, full product chat execution, synthesis, PPA, FPGA
work, U280 work, replay, or mutation of consumed evidence. The current
checksum-valid control-plane record binds the accepted nonofficial sidecar
integration and explicitly denies the next official RTL-backed arbitrary-text
chat step; sealed Fresh review of mission `4fdeed5b2c4a` also records that the
self-labeled Manager authorship of that transition is not independently proven.
Stage 1 advancement therefore remains closed by the authority gate described
below. The stale `BLOCKED_EXTERNAL_ATTESTATION` wait was resolved only for
bounded, nonofficial, software-diagnosis/review work; that does not weaken the
functional product criteria and does not convert package readiness, diagnosis,
focused equivalence, or sidecar-integration evidence into a chat, RTL-product,
or deployment result.

Planner cycle 4 records one supervisor-conditioned exception after Fresh-L2
mission `32444a7fc5ab` returned `done` for the frozen rank-1 integer candidate:
an isolated, non-shell-admitted rank-1 V-correction RTL/reference microarchitecture
may be built and checked against the frozen candidate hashes only. The exception
does not alter the public `ace2_shell` interface, does not mutate the certified
attempt-0002 baseline, and does not authorize official execution, `attempt-0003`,
synthesis/PPA, U280/XRT, Stage 2, or product-completion claims. Its specific
interface and acceptance matrix are recorded in
`design/STAGE1_LAYER23_V_RANK1_INTEGER_CORRECTION_SPEC.md`.

Planner cycle 7 records Fresh-L2 mission `271efb2d9974` returning `done` for the
descriptor-bound layer-23 V rank-1 sidecar integration. The reviewer handoff
hashes to `52b06dcbbd21e00131582bc2deea7c35bae64da53fb9dbdaf979961c508f1fa9`
and independently reproduces integration result
`evidence/verification/stage1-layer23-v-rank1-integration-v1/latest/RESULT.json`
SHA-256 `36496c88ea1192a5d22f5b9e3593ddc556ceb4f229bdfc6c3444191c555e23a1`.
The review accepts default-disabled, exact-descriptor, V-phase-only shell
integration with zero new shell-lint diagnostics beyond the preserved baseline
warning set. It creates no official arbitrary-text chat claim, official
preflight/run, `attempt-0003`, synthesis/PPA, U280/XRT action, or product
completion.

### Stage 1 cycle-level prefill, KV, and decode behavior

The local demo accepts arbitrary natural-language UTF-8 text through stdin,
through a caller-supplied prompt file, or through a caller-supplied UTF-8 JSON
`--conversation-file` whose roles alternate user/assistant and end in user. The tokenizer applies the frozen local
Qwen2.5-0.5B-Instruct chat template selected by the runtime package; malformed
UTF-8, empty prompts after template construction, out-of-range token IDs, or
package-hash mismatches fail closed before command execution.

| Step | Cycle-level behavior |
| --- | --- |
| Command acceptance | The host presents at most one direct `ace2_shell` command while `cmd_ready_o` is high. A command is accepted only on a rising `clk_i` edge with `cmd_valid_i && cmd_ready_o`; every command field is sampled on that edge and remains the command contract until completion. |
| Prefill step `i` | For prompt token position `i`, the runtime issues the complete frozen command grammar for all transformer layers in order: input RMSNorm, Q/K/V projections, RoPE, causal score, softmax, attention value/compose, output projection, residual, post-attention RMSNorm, gate/up projections, SiLU-gate, down projection, final residual, final RMSNorm, and LM-head tile sequence when required by the package. The default product schedule uses one fused opcode `0x0b` QKV command except at layer-0 prompt-prefill positions carrying authenticated Dynamic Scale32 sidecars, which preserve the three opcode `0x01` Q/K/V commands. |
| KV visibility | A token position is not complete until every layer's K/V payload, metadata, destination digest, completion tag, saturation/error state, and all write responses have retired. Position `i+1` and any decode position may only consume K/V records whose producer completion is journal-authenticated. |
| First decode decision | After the final prompt position completes, the RTL-backed LM-head tile sequence produces greedy logits and one token ID. The host may feed this token back to the tokenizer/generation loop only after the independent fixed-point reference agrees on every command destination, LM-head tile value, and the argmax token. |
| Decode feedback step `j` | Generated token position `j` runs the same ordered command grammar with causal context covering all prompt and prior generated positions. The package may stop early only when the generated token is one of the frozen terminators; otherwise it continues until the caller-selected `max_new_tokens` limit. |
| Termination | The demo must emit at least two visible non-termination tokens for a success claim unless a future contract explicitly changes the minimum. A first-token terminator is a boundary case and records no successful arbitrary-text readable-output claim. |
| Completion | Successful completion records prompt token IDs, generated token IDs, decoded text, command count, per-command cycle records, wall time, peak RSS, source-read and destination-write digests, reference-agreement status, and immutable hashes. Any first mismatch or command error stops at the exact boundary and records no Stage 1 completion claim. |

### Input, token, and generation legal ranges

| Item | Legal range and interpretation |
| --- | --- |
| Prompt encoding | Strict UTF-8; the final user message must contain at least one non-whitespace character and tokenize to at least one user-content token |
| Conversation roles | Optional system message, then alternating user/assistant messages beginning with user and ending with user |
| Model token ID | Unsigned integer `0..151935`; any tokenizer or feedback value outside the 151,936-row embedding/LM-head domain fails closed |
| Prompt positions | At least one templated token; absolute sequence positions are `0..32767` |
| `max_new_tokens` | Caller-selected integer `1..256` |
| Total scheduled positions | Prompt-token count plus requested generation positions must not exceed 32,768 |
| Frozen terminators | Token IDs `151643` and `151645` |
| Successful visible output | At least two decoded non-termination tokens; empty, whitespace-only, unmapped, or unreadable decode is not success |

### Clock, reset, flow-control, latency, and throughput rules

The accelerator has one clock domain, `clk_i`. No internal CDC is permitted.
The current public `ace2_shell` contract is single-domain. `rst_ni` is active low. Assertion may
occur asynchronously, but Deassertion at the accelerator boundary must be
synchronous to `clk_i` before state machines leave reset. Reset clears accepted
but incomplete commands, provisional memory publications, completion state,
journal-append eligibility, KV-valid state, and argmax state; authenticated
completed journal prefixes may be replayed only by the host runtime rules.

Every ready/valid channel transfers only on `valid && ready` at a rising
`clk_i` edge. A source holds valid and its complete payload stable while ready
is low. The external-memory contract permits the earliest response one cycle
after request acceptance; same-cycle combinational response is unsupported.
Arbitrary finite backpressure is legal on CSR, direct-command completion,
external-memory request/write/read/response, and SRAM request/read channels.
Backpressure may increase wall time and cycle count but must not reorder,
drop, duplicate, or mutate payloads, tags, completion fields, or K/V records.

The shell accepts at most one direct command at a time; there is no finite
latency guarantee for a command, token, or prompt because legal memory and sink
backpressure may be arbitrarily long. Timeouts are test/runtime failures, not
architectural latency bounds. The demo records measured wall time, simulator
cycles, and command cycles, but no fixed cycles/token or tokens/s value is
claimed by the specification.

### Direct-command control and legal-value rules

All command fields are unsigned packed bits at the public interface. The shell
interprets them using the exact descriptor rules below. All listed command
addresses are byte addresses and must be 16-byte aligned. Values outside these
rules are illegal descriptors and must not begin a successful payload
completion.

| Opcode | Operation | Legal `m/n/k`, layer, position, and flag interpretation |
| ---: | --- | --- |
| `0x01` | W4A8 projection / LM-head tile | Transformer layers `0..23`: `m=1..16` and `(k,n)` in `{(896,896),(896,128),(896,4864),(4864,896)}`. Final LM-head layer `24`: `m=1,n=32,k=896`. Flag bit 6 is legal only for the exact authenticated layer-0 dynamic-Scale32 Q/K/V tuples. |
| `0x02` | RMSNorm | `m=1,n=896,k=0,layer=0..24`. Flag bit 6 is legal only for the exact authenticated layer-0 dynamic-Scale32 RMSNorm tuple. |
| `0x03` | RoPE | `m=1,n` in `{896,128}`, `k=0`, layer `0..23`, sequence position `0..32767`, and flags exactly `45` (`0x2d`) for the maintained basis contract. |
| `0x04` | Attention score | `m=1,n=1..8,k=64`, layer `0..23`; context longer than eight tokens is decomposed into ordered runtime tiles. |
| `0x05` | Softmax | `m=1,n=1..8,k=0`, layer `0..23`. |
| `0x06` | Attention value | `m=1,n=1..8,k=64`, layer `0..23`. |
| `0x07` | SiLU-gate | `m=1,n=1..4864,k=0`, layer `0..23`. |
| `0x08` | Residual add | `m=1,n=896,k=0`, layer `0..23`. |
| `0x09` | Attention compose | `m=1,n=1..8,k=64`, layer `0..23`; flags are the ordered compose phase commands `0x00..0x06`. |
| `0x0a` | K/V publication | `m=1,n=128,k=0`, layer `0..23`, sequence position `0..32767`. |
| `0x0b` | Fused QKV projection | `m=1,n=896,k=896`, layer `0..23`, flags `0`. One descriptor emits ordered Q(896), K(128), and V(128) outputs with shared activation reads. |

`cmd_completion_tag_i` is any unsigned 16-bit value and is returned unchanged
on the corresponding successful completion. Requests are serialized: a second
direct command is not accepted while `busy_o=1`. The runtime package, not the
operator prompt text, selects addresses, flags not constrained above, and the
ordered command expansion. No hidden software operator may replace a required
RTL command.

For opcode `0x0b` at layer `L` in `0..23`, the only legal address tuple is
`src0=0x0000001000000700`,
`src1=0x0000000100000000 + L*0x000000000071c000`,
`scale=0x0000000200000000 + L*0x0000000000031800`,
`dst=0x0000001000000a80`, and `scratch=0`. The `src1` span is exactly
516096 bytes and contains contiguous Q, K, and V signed-W4 weights at byte
offsets `0`, `401408`, and `458752`. The `scale` span is exactly 18432 bytes
and contains their 16-byte per-channel metadata at offsets `0`, `14336`, and
`16384`. The shell reads the 896-byte signed-int8 `src0` tile as 56 packed
128-bit beats once and reuses it for all three projections. The 1152-byte
`dst` span receives Q, K, and V int8 outputs at offsets `0`, `896`, and
`1024`.

Alignment and in-image range are necessary but not sufficient: an aligned
weight, metadata, activation, or destination base other than the exact tuple
above is an illegal descriptor and must be rejected before command execution.
One completion is returned only after all 1152 output bytes retire, and its
saturation/error status is the aggregate of all three projections. Reset
invalidates an in-progress fused transaction. Opcode `0x01` descriptors and
their memory layout remain unchanged.

### Final LM-head tile descriptor and traffic-equivalence binding

For the final LM-head tile, opcode `0x01` is legal only at layer `24` with
`m=1,n=32,k=896`; transformer-layer projection shapes and fused-QKV descriptors
must not be reinterpreted as LM-head tiles. The public focused LM-head harness
checks two 32-logit tiles over the 151,936-row vocabulary tiling (`tiles=4748`)
without creating a new full-vocabulary runtime measurement.

Fresh-L2 accepted the descriptor-bound successor against the immutable
7168-activation/7168-weight-read baseline in mission `stage1lmheadequiv01`.
The accepted result is
`evidence/verification/lm-head-descriptor-traffic-equivalence-v1/reviewer-acceptance-0001/result.json`,
SHA-256
`b10e69c90bffe512fb54153c93a51db0d19fb3f7c4d74f12da96c770f09835e5`; the
Engineer result SHA-256 is
`7a9009a51084871836be78278f96ce9b16bf1d88c1407996b00e65b25ddd2408`.
The accepted successor reads, per functional checked case, 1,792 activation
beats, 896 packed-W4 128-bit beats, and 32 metadata beats, writes two 128-bit
beats, spans exactly 14,336 LM-head packed-weight bytes, and reaches high-water
weight offset 14,320. The predecessor baseline reads 7,168 activation beats,
7,168 W4 beats, and 32 metadata beats.

The accepted equivalence proof reconstructs every one of 28,672
baseline-plus-successor pair records from public vector beats and address
formulas; each trace has 14,336 ordered MAC pair handshakes, 57,344 logical A8
byte occurrences, 57,344 W4 nibble occurrences, no gaps or duplicates, lane 0
in bits `[7:0]`, and even/odd input channels mapped to low/high W4 nibbles.
All 64 checked logits, accumulators, saturation indications, six command
accepts/completions, and ordered completion tags `0x6200..0x6205` match
excluding the intended cycle reduction. The accepted full-runtime tokens `[0,0]`
are preserved structurally by the all-address/all-lane mapping and unchanged
projection arithmetic under the frozen address-to-data memory contract; this is
not a sealed-runtime retry or a new product-quality claim.

This binding creates no hidden harness/golden access, official namespace,
attempt replay, sealed/certified mutation, RTL restoration, synthesis, Yosys
mapping, OpenSTA, PPA, timing optimization, Stage 2, FPGA, or U280 claim. The
focused public harness directly measures two 32-logit tiles, not a complete
arbitrary-text chat result.

### Active layer-0 Dynamic Scale32 contract

Flag value `0x49` selects the active Dynamic Scale32 path only for the four
exact tuples below. No other opcode, layer, shape, flag value, or address
combination may use flag bit 6 successfully. Before any operand payload is
issued, the shell reads four ordered 16-byte beats from `src0_addr-64` and
validates one 64-byte sidecar. The payload address must be at least 64 and
64-byte aligned.

| Producer/consumer | Exact command tuple | Exact addresses | Required input-sidecar identity |
| --- | --- | --- | --- |
| Initial RMSNorm | opcode `0x02`, flags `0x49`, layer 0, `m/n/k=1/896/0` | `src0=0x0000001000000000`, `src1=0`, `dst=0x0000001000000700`, `scale=0x0000000300000000`, `scratch=0` | producer tag equals `cmd_sequence_position_i`; layer `0xff`; producer opcode `0xfe` |
| Q projection | opcode `0x01`, flags `0x49`, layer 0, `m/n/k=1/896/896` | `src0=0x0000001000000700`, `src1=0x0000000100000000`, `dst=0x0000001000000a80`, `scale=0x0000000200000000`, `scratch=0` | producer tag is `(cmd_completion_tag_i-1) mod 2^16`; layer 0; producer opcode `0x02` |
| K projection | opcode `0x01`, flags `0x49`, layer 0, `m/n/k=1/128/896` | `src0=0x0000001000000700`, `src1=0x0000000100062000`, `dst=0x0000001000000e00`, `scale=0x0000000200003800`, `scratch=0` | producer tag is `(cmd_completion_tag_i-2) mod 2^16`; layer 0; producer opcode `0x02` |
| V projection | opcode `0x01`, flags `0x49`, layer 0, `m/n/k=1/128/896` | `src0=0x0000001000000700`, `src1=0x0000000100070000`, `dst=0x0000001000000e80`, `scale=0x0000000200004000`, `scratch=0` | producer tag is `(cmd_completion_tag_i-3) mod 2^16`; layer 0; producer opcode `0x02` |

The sidecar is little-endian and byte-exact:

| Bytes | Required value |
| ---: | --- |
| `0..3` | ASCII `BFP1` (`0x31504642` as the packed little-endian word) |
| `4` | Schema `1` |
| `5` | Group lanes `128` |
| `6` | Group count `7` |
| `7` | Zero |
| `8..9` | Producer tag, little-endian |
| `10` | Producer layer identity from the tuple table |
| `11` | Producer opcode identity from the tuple table |
| `12..15` | Tensor element count `896`, little-endian |
| `16..23` | Model identity `0xadcc20785542c188`, little-endian |
| `24..30` | Seven signed two's-complement exponent deltas, one per 128-lane group |
| `31..63` | Zero; bytes `31..61` are unused delta capacity and bytes `62..63` are reserved |

Validation uses seven immutable base Scale32 records `0x00008000`: reserved
byte zero, normalized significand `0x8000`, and exponent zero. A generic delta
must be in `[-24,+24]`, and base exponent plus delta must be in `[-24,+4]`;
therefore this fixed live path accepts only deltas `[-24,+4]`. Any sidecar
magic, schema, shape, producer tag/layer/opcode, element count, model identity,
zero-fill, address, base-Scale32, delta, or effective-exponent mismatch fails
before operand payload issue. Structural mismatches are descriptor errors;
delta/effective-exponent failures are numeric-overflow errors.

For Q/K/V consumption, group `proj_group_idx_q[7:5]` selects the delta. A
nonnegative delta left-shifts the signed int8 payload value; a negative delta
right-shifts its magnitude with round-to-nearest, ties-to-even, then restores
the sign. Any result outside `[-128,127]` terminates with numeric overflow
before projection weight processing for that group. These projection commands
do not generate an output sidecar.

For RMSNorm output, group `out_idx_q[5:3]` selects the delta. A negative delta
left-shifts the signed int8 core result, zero leaves it unchanged, and a
positive delta right-shifts with the same ties-to-even rule. Any result outside
the symmetric mantissa range `[-127,127]`, or any RMSNorm core saturation,
prevents successful completion and prevents output-sidecar publication.
Payload writes performed before a later numeric failure remain provisional and
must not be journal-authenticated by the host.

After all 896 RMSNorm payload elements and the core completion succeed without
saturation or Dynamic Scale32 overflow, the shell writes four ordered 16-byte
beats of a generated sidecar to `dst_addr-64` before command completion. Its
bytes are the same fixed layout above, with tag=`cmd_completion_tag_i`, layer
0, opcode `0x02`, element count 896, model identity
`0xadcc20785542c188`, and bytes `24..30` copied from the validated input
sidecar; every other unspecified byte is zero. Reset or soft reset clears the
buffered sidecar and pending Dynamic Scale32 overflow state.

### Reset-observable values

While `rst_ni=0`, `csr_ready_o=1`; CSR response valid/data/error, `irq_o`,
`cmd_ready_o`, `busy_o`, command completion valid/tag/error/sum/inverse-RMS/
saturation, external-memory request/write valid, external response ready, and
all SRAM request-valid bits are zero. After synchronous boundary deassertion,
the host must enable the shell through CSR control before the first command is
accepted. Reset during a command discards its provisional state and produces
no successful completion for that command.

### Current implemented CSR protocol and register map

The current shell uses a 32-bit byte address, 64-bit data, and eight byte
strobes. Exactly one request may be pending and one response may be held. A
request transfers on `csr_valid_i && csr_ready_o`; its registered response is
not combinational in the acceptance cycle. `csr_rvalid_o`, `csr_rdata_o`, and
`csr_error_o` remain stable until `csr_rready_i` accepts the response. A write
response returns the pre-write read value of the addressed register. Address
decode requires `csr_addr_i[31:8]==0` and an exact implemented low-byte offset.
An unknown address returns zero with `csr_error_o=1`; when
`CONTROL.strict_errors=1`, it also sets `ERROR_STATUS.reserved_csr` and
interrupt-status bit 0. Writes to known read-only registers are ignored and do
not raise `csr_error_o`.

Only byte strobe 0 controls `CONTROL`, `ERROR_STATUS`, `INTERRUPT_ENABLE`, and
`INTERRUPT_STATUS`; if it is zero, those registers do not change. `DESC_HEAD`
uses strobes 3:0, and full-width writable registers honor all eight strobes.
All stateful CSR fields hard-reset to zero unless the table gives a constant
read value.

| Offset | Name | Access and read value | Implemented write/side effect |
| ---: | --- | --- | --- |
| `0x000` | `ID` | RO constant `0x4143453200000001` | Ignored |
| `0x008` | `VERSION` | RO constant `0x0000000000000001` | Ignored |
| `0x010` | `CAPABILITIES` | RO constant `0x0000000000081091` | Ignored |
| `0x018` | `CONTROL` | RW, low five bits | Bit 0 enables direct-command acceptance; bit 1 requests one soft-reset pulse and self-clears; bit 2 globally enables `irq_o`; bit 3 makes any sticky error halt new commands; bit 4 clears all four performance counters and self-clears |
| `0x020` | `STATUS` | RO snapshot | Bit 0 `idle=!busy_o`; bit 1 `busy=busy_o`; bit 2 `halted_on_error=CONTROL.strict_errors && ERROR_STATUS!=0`; bit 3 is exactly `!cmd_valid_i`; bit 4 is watchdog active; remaining bits are zero |
| `0x028` | `ERROR_STATUS` | W1C sticky low six bits | Bit 0 descriptor, bit 1 memory/response, bit 2 numeric, bit 3 watchdog, bit 4 soft-reset-while-busy, bit 5 reserved CSR; writing one clears the corresponding bit |
| `0x030` | `INTERRUPT_ENABLE` | RW low six bits | Enables corresponding `INTERRUPT_STATUS` bits before global gating |
| `0x038` | `INTERRUPT_STATUS` | W1C sticky low six bits | Bit 0 is shared by successful completion, descriptor failure, and strict unknown-CSR notification; bits 1..4 are memory, numeric, watchdog, and reset-busy; bit 5 has no hardware set source in this implementation |
| `0x040` | `DESC_BASE` | RW 64-bit compatibility register | Byte-strobed storage only; it is not consumed by the current direct-command dispatch path |
| `0x048` | `DESC_HEAD` | RW unsigned low 32 bits, upper read bits zero | Strobes 3:0 update the low 32 bits; arithmetic wraps modulo 2^32 |
| `0x050` | `DESC_TAIL` | RO unsigned low 32 bits, upper read bits zero | Increments modulo 2^32 on the shell's implemented terminal-retirement pulses; it is not an autonomous descriptor-ring consumer pointer |
| `0x058` | `DOORBELL` | WO; reads return zero | Any write increments `DESC_HEAD` by one modulo 2^32, independent of write data and strobes; it does not fetch or dispatch a descriptor |
| `0x060` | `WATCHDOG_LIMIT` | RW 64-bit cycle limit | Zero disables the watchdog; a nonzero limit is reloaded on forward progress and times out an active non-completion command after the implemented countdown expires |
| `0x068` | `PERF_CYCLE` | RO wrapping 64-bit count | Increments once for each registered cycle event sampled while `CONTROL.enable=1` |
| `0x070` | `PERF_BYTE` | RO wrapping 64-bit count | Adds 16 once when the registered event reports an accepted external read response or accepted external write-data beat |
| `0x078` | `PERF_TOKEN` | RO; wrapping count in bits 31:0, bits 63:32 remain zero | Adds projection `m` on a qualifying projection retirement and one on each other qualifying core/compose retirement |
| `0x080` | `PERF_STALL` | RO wrapping 64-bit count | Increments once for a registered cycle containing any external request, write-data, read-response, or write-response stall predicate |

`irq_o` is a level output equal to
`CONTROL.irq_global_enable && |(INTERRUPT_STATUS & INTERRUPT_ENABLE)`. Sticky
error and interrupt causes clear only by hard reset, soft reset, or their W1C
writes. A soft reset clears in-flight command/provisional/completion state and
the error/interrupt status registers; if the shell was busy, bit 4 is then set
in both status registers. It does not reset `DESC_BASE`, `DESC_HEAD`,
`DESC_TAIL`, `INTERRUPT_ENABLE`, or `WATCHDOG_LIMIT`. It does not reset the
performance counters unless `CONTROL.perf_clear` is asserted in the same or
another control write.

The four `DESC_*`/`DOORBELL` registers above are compatibility/diagnostic
state only in this direct-command implementation. There is no CSR-driven
descriptor fetch engine, ring-size register, last-result register, or
last-error-info register in the active RTL. Historical ring semantics are not
part of this frozen contract.

### Public `ace2_shell` parameter and port contract

The live runtime instantiates the following complete public parameter set.
Parameters are signed SystemVerilog `integer` elaboration constants, but this
implementation supports exactly the tuple below; no independent override or
alternate combination is legal for the local-Qwen runtime. An unsupported tuple
is outside the interface contract rather than a smaller or larger supported
configuration.

| Parameter | Default | Legal value | Contract |
| --- | ---: | ---: | --- |
| `HIDDEN_SIZE` | `ACE2_HIDDEN_SIZE` (896) | 896 only | Hidden-vector element count |
| `LANES` | `ACE2_VECTOR_LANES` (16) | 16 only | Elements per 128-bit activation beat |
| `ACT_WIDTH` | 8 | 8 only | Signed activation and memory-lane semantic width |
| `GAIN_WIDTH` | 16 | 16 only | Signed RMSNorm gain semantic width |
| `ACC_WIDTH` | 48 | 48 only | RMSNorm sum-of-squares completion width |
| `INV_RMS_FRAC` | 30 | 30 only | Fraction bits in the reciprocal-RMS result |
| `GAIN_FRAC` | 8 | 8 only | Fraction bits in RMSNorm gains |
| `PROJ_MAC_LANES` | 4 | 4 only | Parallel W4A8 projection MAC lanes |
| `PROJ_ACC_WIDTH` | 32 | 32 only | Signed projection accumulator semantic width |
| `PROJ_M_MAX` | 16 | 16 only | Maximum projection row count accepted by the shell |
| `SRAM_BANKS` | `ACE2_SRAM_BANKS` (8) | 8 only | Number of public SRAM request banks |
| `SRAM_ADDR_WIDTH` | `ACE2_SRAM_ADDR_WIDTH` (12) | 12 only | Per-bank SRAM byte-address width |
| `ENABLE_LAYER23_V_RANK1_SIDECAR` | 0 | 0 or 1 only | Qualifies the exact layer-23 V fused-QKV command match and binds to `ace2_layer23_v_rank1_integer_correction_sidecar.ENABLE_SIDECAR`; default 0 preserves byte-identical baseline shell behavior |
| `LAYER23_V_RANK1_CONFIG_VALID` | 1 | 0 or 1 only | Gates an enabled exact-match command fail closed before memory traffic and binds to `ace2_layer23_v_rank1_integer_correction_sidecar.CONFIG_VALID` |

Every public port below is declared as an unsigned SystemVerilog scalar or
packed bit vector; there are no `signed` public port declarations. Signed
numeric meaning is applied only when an operator interprets payload lanes or a
completion field as stated in its numerical contract.

| Port | Direction | Width (unsigned packed bits) | Contract |
| --- | --- | --- | --- |
| `clk_i` | input | 1 | Rising-edge clock |
| `rst_ni` | input | 1 | Asynchronous active-low reset |
| `csr_valid_i` | input | 1 | CSR request valid |
| `csr_ready_o` | output | 1 | CSR request ready |
| `csr_write_i` | input | 1 | CSR write when one, read when zero |
| `csr_addr_i` | input | 32 | CSR byte address |
| `csr_wdata_i` | input | 64 | CSR write data |
| `csr_wstrb_i` | input | 8 | CSR byte strobes |
| `csr_rvalid_o` | output | 1 | CSR response valid; held until ready |
| `csr_rready_i` | input | 1 | CSR response ready |
| `csr_rdata_o` | output | 64 | CSR read data |
| `csr_error_o` | output | 1 | CSR response error |
| `irq_o` | output | 1 | Level interrupt output |
| `cmd_valid_i` | input | 1 | Direct command valid |
| `cmd_ready_o` | output | 1 | Direct command ready |
| `cmd_opcode_i` | input | 8 | Command opcode |
| `cmd_flags_i` | input | 8 | Opcode-specific flags |
| `cmd_layer_id_i` | input | 8 | Transformer layer, including final layer 24 |
| `cmd_m_i` | input | 16 | Command row/count dimension |
| `cmd_n_i` | input | 16 | Command output/vector dimension |
| `cmd_k_i` | input | 16 | Command reduction/context dimension |
| `cmd_sequence_position_i` | input | 16 | Absolute zero-based token position |
| `cmd_completion_tag_i` | input | 16 | Completion tag returned unchanged |
| `cmd_src0_addr_i` | input | 64 | First operand byte address |
| `cmd_src1_addr_i` | input | 64 | Second operand byte address |
| `cmd_dst_addr_i` | input | 64 | Destination byte address |
| `cmd_scale_addr_i` | input | 64 | Scale/metadata byte address |
| `cmd_scratch_addr_i` | input | 64 | Scratch, KV, or auxiliary byte address |
| `mem_req_valid_o` | output | 1 | External-memory request valid; held until ready |
| `mem_req_ready_i` | input | 1 | External-memory request ready |
| `mem_req_write_o` | output | 1 | Request type: write when one |
| `mem_req_addr_o` | output | 64 | External byte address |
| `mem_req_len_o` | output | 16 | Request length in 128-bit beats |
| `mem_req_tag_o` | output | 8 | Request/response correlation tag |
| `mem_wvalid_o` | output | 1 | External write-data valid; held until ready |
| `mem_wready_i` | input | 1 | External write-data ready |
| `mem_wdata_o` | output | `LANES*ACT_WIDTH` | External write payload |
| `mem_wstrb_o` | output | 16 | External write byte strobes |
| `mem_wtag_o` | output | 8 | External write-data tag |
| `mem_rvalid_i` | input | 1 | External read-response valid; source holds payload until ready |
| `mem_rready_o` | output | 1 | External read-response ready |
| `mem_rdata_i` | input | `LANES*ACT_WIDTH` | External read payload |
| `mem_rtag_i` | input | 8 | External read-response tag |
| `mem_rerror_i` | input | 1 | External read-response error |
| `mem_bvalid_i` | input | 1 | External write-response valid; held until ready |
| `mem_bready_o` | output | 1 | External write-response ready |
| `mem_btag_i` | input | 8 | External write-response tag |
| `mem_berror_i` | input | 1 | External write-response error |
| `sram_req_valid_o` | output | `SRAM_BANKS` | Per-bank SRAM request valid |
| `sram_req_ready_i` | input | `SRAM_BANKS` | Per-bank SRAM request ready |
| `sram_write_o` | output | `SRAM_BANKS` | Per-bank SRAM write select |
| `sram_addr_o` | output | `SRAM_BANKS*SRAM_ADDR_WIDTH` | Packed per-bank SRAM byte addresses |
| `sram_wdata_o` | output | `SRAM_BANKS*LANES*ACT_WIDTH` | Packed per-bank SRAM write payloads |
| `sram_wstrb_o` | output | `SRAM_BANKS*16` | Packed per-bank SRAM byte strobes |
| `sram_rdata_i` | input | `SRAM_BANKS*LANES*ACT_WIDTH` | Packed per-bank SRAM read payloads |
| `sram_rvalid_i` | input | `SRAM_BANKS` | Per-bank SRAM read valid |
| `busy_o` | output | 1 | A command is active |
| `cmd_done_valid_o` | output | 1 | Completion valid; held until ready |
| `cmd_done_ready_i` | input | 1 | Completion ready |
| `cmd_done_tag_o` | output | 16 | Completed command tag |
| `cmd_done_error_o` | output | 1 | Terminal command error |
| `cmd_done_sumsq_o` | output | `ACC_WIDTH` | RMSNorm sum of squares, zero when not applicable |
| `cmd_done_inv_rms_q30_o` | output | `INV_RMS_FRAC+2` | Reciprocal RMS result, Q30 at the default parameter |
| `cmd_done_saturation_seen_o` | output | 1 | Any specified output saturation occurred |

### Mission-specific acceptance matrix

The matrix below is a specification contract only. It does not authorize
downstream execution while the pipeline remains in `specification`.

| Class | Required case | Observable acceptance |
| --- | --- | --- |
| Normal | Fresh natural-language prompt through the local demo path | All prompt tokens prefill, K/V grows per position, at least two visible generated non-termination tokens decode, and every accelerator/reference boundary is bit-exact |
| Boundary | One-token user text, caller cap of one generated token, first-token terminator, first prompt token, first generated token, and maximum legal `max_new_tokens` value | Correct token counts, correct no-extra-token behavior, legal range enforcement, and explicit no-success claim when the visible-token minimum is not met |
| Illegal | Empty prompt, malformed UTF-8, digest mismatch, illegal descriptor including an aligned but noncanonical fused-QKV base, invalid token/count, wrong memory response tag, or memory error | Fail closed before consumption or stop at the first command boundary with no successful Stage 1 claim |
| Reset | Reset during prefill, decode, K/V publication, LM-head tile, and completion response | No stale valid, K/V, argmax, or provisional destination state remains after reset release |
| Stall | CSR, command, memory request, write data, read response, write response, SRAM, and completion backpressure | Payloads and tags remain stable; no loss, duplication, reordering, or premature completion occurs |
| Recovery | Resume from an authenticated journal prefix immediately before and after command and token boundaries | Completed prefix authenticates; resumed output, K/V, logits, token IDs, and decoded text equal uninterrupted execution |
| Preservation | Existing certified two-token records and sealed evidence | No replay, mutation, relabeling, or new correctness conclusion is made from sealed predecessor evidence |

### Stage 1 acceptance matrix

| Class | Required case | Observable acceptance |
| --- | --- | --- |
| Normal | Fresh arbitrary natural-language input with `max_new_tokens` in the legal range | Readable multi-token output with at least two visible non-termination tokens, measured wall time, command cycles, generated token IDs, and exact RTL/reference agreement |
| Boundary | Empty/one-token prompt boundary, `max_new_tokens` values 1 and 256, first-token terminator, and earliest legal memory response | Fail-closed or exact documented boundary result with no hidden fallback and no unsupported latency claim |
| Illegal | Malformed UTF-8, invalid package hash, out-of-range token ID, illegal descriptor including an aligned but noncanonical fused-QKV base, wrong response tag, memory error, or reference mismatch | Stop at the first observable boundary, preserve evidence, and emit no Stage 1 completion claim |
| Reset | External reset or soft reset before, during, and after prefill/decode command windows | Cleared provisional state, no stale K/V or argmax visibility, and recovery only from an authenticated completed prefix |
| Stall | Arbitrary finite backpressure on every ready/valid channel | Stable valid/payload behavior, no data loss, no duplication, no reordering, and completion only after all required responses retire |
| Recovery | Interrupted/resumed run at command and token boundaries | Journal authentication succeeds and resumed commands, K/V, logits, tokens, decoded text, and reference digests match an uninterrupted run |

### Stage 1 current authority and advancement gate

The functional acceptance matrix above is necessary but is not currently
authorized for a product execution or completion claim. Live Planner-cycle-8
inspection found the Manager-owned `research/PIPELINE_STATE.json` at SHA-256
`9971288788631bbf12b99020e9364523a43b3333b0a7336a34c07382f1be5936` and
`current_stage=stage1_software_diagnosis`; the checksum companion records the
same SHA-256. The sealed Fresh review for mission `4fdeed5b2c4a` records that
the transition labeled `by=manager` was inserted during an Engineer call, so
independent Manager replacement or explicit ratification is still required for
the provenance requirement. This Planner cycle did not edit
`research/PIPELINE_STATE.json` or its checksum companion.

The live `research/PIPELINE_STATE.json#stage1_blocker` records status
`NONOFFICIAL_RTL_REFERENCE_INTEGRATION_ACCEPTED_OFFICIAL_CHAT_DENIED`, with
previous status `BLOCKED_EXTERNAL_ATTESTATION`. The operator-answer decision and
later bounded work advanced only nonofficial Stage 1 diagnosis, candidate,
focused RTL/reference, generalization, and descriptor-bound sidecar-integration
evidence. That authority is consumed and the current record explicitly denies
the next official RTL-backed arbitrary-text chat step:
`bounded_diagnosis_authority_consumed=true`,
`bounded_nonofficial_software_diagnosis_authorized=false`,
`checkpoint_176_official_execution_authorized=false`,
`fresh_l2_review_authorized_for_bounded_diagnosis=false`,
`official_rtl_backed_arbitrary_text_chat_authorized=false`,
`rtl_authorized=false`, `stage_2_authorized=false`, `xrt_authorized=false`, and
`u280_authorized=false`. The live next action is
`DENY_NEXT_OFFICIAL_RTL_BACKED_ARBITRARY_TEXT_CHAT`; it does not authorize
official preflight/run, `attempt-0003`, retry/replay/resume, checkpoint-176
official execution, W4/RTL promotion, full Icarus generation as product
evidence, synthesis, PPA, Stage 2, proprietary-toolchain installation, XRT,
FPGA, U280, or product completion.

The Manager-owned record binds prior Q/K/V diagnosis result
`diagnosis/stage1layer23qkvfamilies01/result.json`, SHA-256
`4e6fa9c101f2d2a4c09a80fb1883665043a5d8126ebaa7d4fa862bf602aa4856`, V
residual-structure diagnosis result
`evidence/diagnostics/stage1-layer23-v-residual-structure-v1/nonofficial-diagnosis-0001/result.json`,
SHA-256 `11dbd7dda4715827e3a16e4e530bdb6870fa51370fff86084a05c6d07ea3bc2d`,
accepted integration result
`evidence/verification/stage1-layer23-v-rank1-integration-v1/latest/RESULT.json`,
SHA-256 `36496c88ea1192a5d22f5b9e3593ddc556ceb4f229bdfc6c3444191c555e23a1`,
manager-reconciliation record
`DENY_NEXT_OFFICIAL_RTL_BACKED_ARBITRARY_TEXT_CHAT` with
`sealed_evidence_replayed=false`, and consumed authorization SHA-256
`87bf833dfcda18d514c366d622a50be834a61422bab7cdf0dab096b3e55f96d9`. Later
supervisor-conditioned nonofficial candidate, RTL, generalization, and
descriptor-bound sidecar work is recorded as project evidence only; the Manager
has not advanced official chat, synthesis/PPA, or Stage 2 authority, and sealed
review keeps Manager-authorship provenance unresolved.

Latest reviewed evidence is bounded and non-product:

| Mission | Fresh-L2 disposition | Product implication |
| --- | --- | --- |
| `9e235ac035b2` | Accepted layer-23 Q/K/V causal diagnosis; V BF16 projection-output substitution with A8 output retained restores token `39814` to rank 1, top-8 overlap 7, JSD `0.112292190479` nats | Diagnosis only; no candidate, RTL, official execution, or Stage 2 authority |
| `stage1vprecision01` | Accepted negative W4A8/W8A8/W4A16/W8A16 V-precision matrix; no format met the material-improvement rule | Do not promote W8A8, W4A16, or W8A16; no current RTL-supported precision mode is added |
| `stage1lmheadequiv01` | Accepted descriptor-bound LM-head traffic equivalence for the packed-beat successor | Focused behavioral equivalence only; no new full-vocabulary runtime, synthesis/PPA, or product-quality claim |
| `stage1vjointscale01` | Accepted bounded nonofficial layer-23 V joint W4/A8 Scale32 search as a supported negative: five candidates tested, zero gate passes, zero held-out no-regression passes, selected candidate `null` | Do not promote a scale-only V candidate; no RTL/reference promotion, official run, PPA, or Stage 2 authority |
| `1421b599ab84` / blocked item `3d3e14c90193` | Accepted terminal closure after the operator explicitly denied advancement or RTL-backed arbitrary-text chat execution authority | The blocked official-chat continuation remains closed; no official preflight/run, `attempt-0003`, attestation, Stage 2, or repository-destructive action |
| `32444a7fc5ab` | Accepted frozen rank-1 integer candidate reproduction, enabling only isolated focused microarchitecture work | No official execution, no shell admission, no synthesis/PPA, no Stage 2, and no product-quality claim |
| `dc94c6bf8107` | Accepted focused rank-1 integer RTL/reference evidence for the isolated core | Focused non-product RTL evidence only; no official attempt, shell integration, synthesis/PPA, or U280 authority |
| `8e6ad3ce7e45` | Accepted four-prompt nonofficial generalization over 290 position vectors with zero catastrophic records | Supports bounded sidecar integration only; no official arbitrary-text chat or product claim |
| `271efb2d9974` | Accepted descriptor-bound layer-23 V rank-1 sidecar integration; result SHA-256 `36496c88ea1192a5d22f5b9e3593ddc556ceb4f229bdfc6c3444191c555e23a1` | Default-disabled, exact-guarded shell integration is reviewed evidence; official preflight/chat, `attempt-0003`, synthesis/PPA, XRT/U280, and product completion remain unauthorized |

No runtime namespace, payload open, evaluator call, target start, B0 effect,
checkpoint/chat admission, RTL authority, downstream-stage authority, XRT
authority, U280 authority, or product-completion claim is created by this
specification/diagnosis record. Stage advancement remains a Manager action;
Planner and Engineer do not edit `research/PIPELINE_STATE.json`; provenance
ratification or replacement must come from the Manager role.

### Benchmark-interface closure

No external accelerator benchmark, hidden harness, hidden golden output, score,
ranking, or third-party performance policy applies to the current local
productization contract. No fixed benchmark attempt is started or
consumed by this specification closure. The local acceptance contract is the
caller-selected `python3 tools/ace2_chat_demo.py --output <dir>` path using the
public `ace2_shell` interface frozen above and independently checked against
the fixed-point reference. Existing sealed two-token evidence remains
predecessor evidence and cannot satisfy the arbitrary-text Stage 1 objective by
itself.

Stage 2 U280 work is ordered after Stage 1 acceptance and a later Manager stage
advance. The latest read-only host inventory is
`research/raw/specification/u280-host-inventory-recheck-20260818T213022Z.json`;
the active host inventory observed no Vivado, Vitis, `v++`, XRT utilities,
authorized hardware-emulation endpoint, matching local PCI function, or XRT
device node. That observation is not evidence about the existence,
availability, identity, functionality, or timing of U280 hardware. No
proprietary toolchain download is authorized by the current
specification-stage work.

## Preserved predecessor: LoRA V4 two-token 24-layer RTL specification

### Status, version, and authority

This preserved predecessor section was the normative specification for mission
`extend-two-token-24layer-rtl-stack`, version `2.0`. It composes the
independently accepted layer-0 V2 operator contract across transformer layers
0 through 23 for exactly two frozen tokenizer positions. Its machine-readable exact interface is preserved under
`preserved_predecessor_two_token_contract` in
`design/BENCHMARK_INTERFACE.json`.

That predecessor specification authorized preview-only preparation and verification. It
does not unlock `evidence/verification/lora-v4-two-token-24layer-rtl-v1/attempt-0001`.
The official namespace remains absent and locked until a later reviewed
contract explicitly changes the gate.

No external benchmark, hidden harness, hidden golden, score, ranking, or
third-party performance contract applies. Acceptance is local, deterministic,
and bit-exact against an independently implemented fixed-point golden. Success
does not establish model quality, final RMSNorm, LM-head execution, token
generation, readable chat, full Stage 1, synthesis/PPA, FPGA/U280 execution,
routed timing, or accelerator latency.

### Frozen identities and bounded input

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, model SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
- Adapter: canonical LoRA V4 checkpoint-176, adapter SHA-256
  `c9476d6ccc687a4d68de1951eb889b689daaf9c5d3822b4c5acfc26a1cd38a28`,
  checkpoint-tree SHA-256
  `8cee9b9343e49b66ee2ea427578f87335ffa259919356cdd119cd8712ff7eeee`.
- Input alias: `canonical_lora_v4_two_position_sequence`. The frozen tokenizer
  sequence is `[5576, 12, 17, 467, 19, 32, 23, 6193, 36929, 29295, 33085, 13]`;
  only positions 0 and 1, token IDs `5576` and `12`, execute. Literal private
  input text is neither needed nor persisted.
- Geometry: 24 layers, hidden size 896, intermediate size 4864, 14 query
  heads, 2 KV heads, head dimension 64, and grouped-query mapping
  `kv_head = query_head // 7`.
- The LoRA merge is
  `base_bf16_as_f32 + (32 / 16) * (lora_B_f32 @ lora_A_f32)` for Q, K, V,
  O, gate, up, and down in every layer.
- The accepted layer-0 `attempt-0001` and `attempt-0002` evidence are immutable
  predecessor evidence and may not be modified, relabeled, or reused as a
  24-layer result.

### Exact per-layer execution and carry schedule

For each layer `L` from 0 through 23, position 0 completes and publishes that
layer's K/V entry before position 1 consumes both the published position-0 K/V
and its new position-1 K/V. Both resulting 896-byte signed-A8 hidden states are
then carried to layer `L+1`. K/V storage is distinct for every layer; no cache
entry, address, or metadata may alias another layer.

The exact operator order in every layer is:

1. input RMSNorm;
2. Q, K, and V W4A8 projections;
3. Q and K RoPE;
4. causal attention score, softmax, and attention compose;
5. O W4A8 projection;
6. attention residual add;
7. post-attention RMSNorm;
8. gate and up W4A8 projections;
9. SiLU-gate product;
10. down W4A8 projection; and
11. final residual add.

Position 0 uses identity RoPE and a singleton causal context. Position 1 uses
non-identity RoPE and context two. In every layer, at least one position-1 head
must assign nonzero probability to position 0, and replacing cached position-0
V with zero must change at least one position-1 attention-compose output.

No final RMSNorm or LM head executes after the layer-23 residual output.

### Exact RTL interface composition

The exact parameter declarations, port declarations, directions, widths,
signed declarations, elaborated values, source hashes, semantic ranges, reset
values, and transaction rules are enumerated directly in
`design/BENCHMARK_INTERFACE.json`. Those declarations are co-normative and are
not merely provenance. No wrapper port, compatibility alias, repaired width,
or signedness reinterpretation is permitted.

The maintained public modules are:

| Family | Module | Exact elaboration for this mission |
| --- | --- | --- |
| Projection | `ace2_w4a8_proj_core` | `K_SIZE` 896 or 4864 as selected by projection, `MAC_LANES=4`, `ACT_WIDTH=8`, `WGT_WIDTH=4`, `ACC_WIDTH=32`; `GROUP_INDEX_WIDTH` follows the declared expression and is 8 or 11 |
| RMSNorm | `ace2_rmsnorm_core` | `HIDDEN_SIZE=896`, `LANES=16`, `ACT_WIDTH=8`, `GAIN_WIDTH=16`, `ACC_WIDTH=48`, `INV_RMS_FRAC=30`, `GAIN_FRAC=8` |
| RoPE | `ace2_rope_core` | `LANES=16`, `ACT_WIDTH=8`, `SCALE_WIDTH=16`, `TRIG_WIDTH=16` |
| Attention score | `ace2_attention_score_core` | `HEAD_DIM=64`, `MAC_LANES=1`, `ACT_WIDTH=8`, `ACC_WIDTH=32`, `SCORE_WIDTH=16` |
| Softmax | `ace2_softmax_core` | `CONTEXT_MAX=8`, `SCORE_WIDTH=16`, `PROB_WIDTH=16` |
| Attention compose | `ace2_attention_compose_core` | `TILE_MAX=8`, `HEAD_DIM=64`, `CONTEXT_MAX=32768`, `ACC_WIDTH=40` |
| SiLU gate | `ace2_silu_gate_core` | no parameters |
| Residual replay | `ace2_shell` | exact 14-parameter, 64-port public shell declaration; `HIDDEN_SIZE=896`, `LANES=16`, `SRAM_BANKS=8`, `SRAM_ADDR_WIDTH=12`; `ENABLE_LAYER23_V_RANK1_SIDECAR` qualifies the exact layer-23 V fused-QKV command match and maps to sidecar `ENABLE_SIDECAR`, while `LAYER23_V_RANK1_CONFIG_VALID` gates that enabled match fail closed before memory traffic and maps to sidecar `CONFIG_VALID` |

For shell residual commands, `cmd_layer_id_i` is exactly the executing layer
`0..23`; opcode is `0x08`, flags are `0x09`, `m=1`, `n=896`, `k=0`, sequence
position is 0 or 1 as applicable, addresses are 16-byte aligned, and the
16-bit completion tag returns unchanged and in order.

### Fixed-point and scaling contract

- Projection weights are signed W4, symmetric per output channel with
  `scale = absmax / 7`; quantization uses round-to-nearest ties-to-even and
  clamps to `[-8, 7]`.
- Operator activations, residual outputs, and K/V cache bytes are signed int8.
  Projection accumulators are complete signed-int32 reductions.
- Requantization uses signed-Q31 multipliers, right shifts in `0..63`, zero
  output zero-point for this workload, ties-to-even rounding, and explicit
  int8 saturation.
- RMSNorm gains are signed Q7.8. RoPE uses theta 1,000,000, signed-Q9
  activation scale one, signed-Q15 cosine/sine coefficients, and split-half
  pairing. SiLU inputs are signed Q6.9.
- Residual adds requantize both inputs to the declared common scale and perform
  signed-int8 saturating addition.
- All per-layer scales, multipliers, shifts, accumulators, outputs, residuals,
  and cache bytes must be emitted for independent comparison. No float-fidelity
  threshold may be tuned or used as a PASS condition.

### Clock, reset, handshake, stall, and recovery

The simulation clock period is 10 ns with a rising accepting edge and 5 ns
half-period. `rst_ni` is active-low and asynchronous in RTL. `clear_i` is tied
low; mid-command clear/reset is outside this bounded claim.

| Family | Reset assertion / settle | Required observable reset state | Required post-reset behavior |
| --- | --- | --- | --- |
| Projection | 4 rising edges / 2 | `start_ready=1`; pair/meta ready and output valid 0; output, accumulator, overflow, saturation 0 | First legal reduction completes; exactly `last_group+1` pair accepts, one metadata accept, one output |
| RMSNorm | 4 / 2 | `start_ready=1`; input/gain/scale ready, output valid, done valid 0; output, sumsq, inverse RMS, saturation 0 | First 56-beat collection plus 56 gain/scale beats completes |
| RoPE | 4 / 2 | `start_ready=1`; beat ready and output valid 0; output and saturation 0 | First exact 56-beat tensor completes |
| Attention score | 4 / 2 | `start_ready=1`; pair ready and output valid 0; score, accumulator, saturation 0 | First exact 64-pair token completes |
| Softmax | 4 / 2 | `start_ready=1`; output valid 0; all probabilities and saturation 0 | First legal context-1 or context-2 case completes |
| Attention compose | 4 / 0 | with zero inputs, `command_allowed=0`, `start_ready=1`; value ready, output valid/last, command done 0; data, saturation, context count 0 | First authorized ordered command sequence completes |
| SiLU gate | 4 / 0 | `start_ready=1`, `beat_ready=1`; output valid/data/saturation 0 | First legal beat sequence completes |
| Residual shell | 5 / 2 | CSR ready 1; command ready, busy, IRQ, completion valid/error/saturation, memory and SRAM request valids 0; completion fields and CSR response 0 | CSR enable precedes the first command; first residual command completes without error |

A source asserts valid only after observing ready, keeps payload stable through
the accepting rising edge, and deasserts on the following falling edge. A sink
initially withholds ready, observes valid, and then accepts. While
`valid=1 && ready=0`, valid and every payload bit must remain stable
indefinitely. Cases execute sequentially without inter-case reset; after output
acceptance each core returns to ready/idle, and shell tags remain ordered.

Random stalls, mid-command reset/clear, and injected memory/tag faults are not
required and produce no claim unless a future version explicitly adds them.
Timeouts are hard failures, not expected latency.

### Legal and illegal controls

Packed fields use two's-complement semantic interpretation where declared.
Signed int4 is `[-8,7]`, signed int8 is `[-128,127]`, signed int16 is
`[-32768,32767]`, signed int32 is `[-2147483648,2147483647]`, and all six-bit
right shifts are `0..63`.

Projection `last_group_i` is `0..1215` and this workload uses 223 or 1215.
Softmax context count is `1..8` and this workload uses 1 or 2. Compose tile
count is `1..8`; legal commands are `0x00..0x06` with the exact MAX, SUM, VALUE
ordering in the interface; `start_authorized_i` must be 1. SiLU lane count is
`1..8`. Shell layer IDs are `0..23` and addresses are 16-byte aligned.

Unsupported compose commands, invalid compose order, zero/excessive counts,
unauthorized compose starts, wrong shell opcode/shape/layer, and misaligned
addresses are illegal. Illegal scenarios are not executed in this version;
their required result is `NOT_EXERCISED_NO_CLAIM`, never silent acceptance.

### Frozen workload and simulator-cycle expectations

Cycle counts are deterministic family-local simulator counts derived from the
accepted layer-0 V2 schedule repeated independently for all 24 layers. They
are not wall-clock or accelerator latency. Cases are serialized within each
family.

| Family | 24-layer accepted transactions | Exact expected cycles |
| --- | ---: | ---: |
| Projection | 178,913,280 pair beats; 608,256 metadata beats; 608,256 outputs | 642,358,128 |
| RMSNorm | 5,376 collect; 5,376 gain/scale; 5,376 output beats; 96 done | 5,977,440 |
| RoPE | 5,376 input/output beat transactions | 18,177,528 |
| Attention score | 64,512 Q/K pair beats; 1,008 score outputs | 135,192 |
| Softmax | 672 starts and 672 outputs | 73,704 |
| Attention compose | 4,032 commands; 24,192 value beats; 2,688 output beats | 816,552 |
| SiLU gate | 29,184 input beats and 29,184 output handshakes | 42,346,152 |
| Residual shell | 96 commands and 5,376 output beats | 86,952 |
| **Aggregate** | 24 serialized layer schedules | **709,971,648** |

Wall-clock duration and peak RSS are measured, recorded, and hash-bound but
are descriptive host observations with no predeclared PASS threshold.

### Scenario-to-observable acceptance matrix

| Class | Required scenario | Observable result |
| --- | --- | --- |
| Normal | All two-position operator paths in layers 0..23 | Zero fixed-point mismatches, all PASS markers, 608,256 exact projection-channel results, and both layer-23 hidden outputs exact |
| Boundary | Position 0 singleton/identity RoPE; position 1 context-two/non-identity RoPE; final/partial packs; layers 0 and 23 | Exact grouped-query mapping, exact packing, correct first/last layer carry, and no cross-layer cache alias |
| Cache dependence | Every layer's position-1 decode | 24/24 nonzero cached-attention contributions and 24/24 cached-V-removal effects |
| Reset | Reset map and first post-reset case in every family | No pre-command valid/error; all frozen reset values match; first legal transaction completes exactly |
| Stall | Deterministic output backpressure | Valid and payload remain stable until acceptance; no loss, duplication, or reordering |
| Recovery | Sequential cases and layer transitions without reset | Core returns ready/idle, shell tags remain exact/in order, both hidden states and each layer's private K/V continue correctly |
| Illegal compose | Not exercised | `NOT_EXERCISED_NO_CLAIM`; if later exercised, invalid command/order/count/authorization must not be accepted |
| Illegal shell | Not exercised | `NOT_EXERCISED_NO_CLAIM`; if later exercised, invalid descriptor must error with no successful payload completion |
| Random stalls | Not exercised | `NOT_EXERCISED_NO_CLAIM` |
| Mid-command clear/reset | Not exercised | `NOT_EXERCISED_NO_CLAIM` |
| Injected memory/tag fault | Not exercised | `NOT_EXERCISED_NO_CLAIM` |

### Acceptance, evidence, and immutable namespaces

One sealed execution passes only if all of the following are true:

1. Every frozen mission, source, checkpoint, predecessor, runner, spec, and
   interface hash verifies before execution.
2. The canonical real LoRA merge and W4 audit covers exactly 168 projections,
   357,826,560 logical weights, and 178,913,280 packed-byte equivalent.
3. Both hidden states cross all 24 layers; each layer emits a distinct 512-byte
   two-position K/V cache, for exactly 12,288 cache bytes total.
4. Independent Python fixed-point versus RTL mismatch count is zero for every
   operator boundary, projection accumulator/output, residual, cache byte, and
   both final 896-byte layer-23 hidden outputs.
5. All 24 cache-dependence and all 24 counterfactual checks pass.
6. Fresh compilation and simulation return codes are zero, exact transaction
   and simulator-cycle totals reproduce, and no timeout or malformed PASS
   marker occurs.
7. Raw logs, vectors, tensors, independent-golden traces, execution sources,
   per-layer cycle counts, total wall time, peak RSS, source hashes, manifest,
   and `SHA256SUMS` are immutable and independently readable by `--check`.

Preview evidence is written only below
`build/lora-v4-two-token-24layer-rtl-preview/`. Existing
`preflight-0001` is immutable predecessor preview evidence. A repaired or
refrozen preview uses the next unused namespace. Preview commands must report
`official_attempt_consumed=false`, and the official attempt path must remain
absent.

`--prepare`, `--run`, and `--check` remain locked by the version-2 contract.
This specification closure alone grants no authority to unlock or consume the
official attempt.

### Failure classification and claim boundary

A missing dependency, private path or literal input leak, hash mismatch,
cross-layer cache alias, compile/simulator failure, timeout, nonzero integer
mismatch, count/cycle mismatch, cache-dependence failure, incomplete immutable
evidence, or official-namespace creation fails the applicable attempt.

Infrastructure or harness failure gives no RTL-correctness conclusion unless a
comparable RTL transaction completed. Success establishes only the exact
two-position transformer stack through layer 23 under this fixed-point and
interface contract.
