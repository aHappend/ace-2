#!/usr/bin/env python3
"""Hash-bind offline token-0 V and replay layer-0 attention value on unchanged RTL."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from ace2_attention_value_reference import AttentionValueCase, reference_attention_value


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-attention-value-offline-v-authority-replay-v1/mission.json"
)
REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-attention-value-offline-v-authority-replay-v1/round-0001.json"
)
OFFLINE = ROOT / "evidence/diagnostics/c4-layer0-attention-value-token0-v-offline-reconstruction-20260803-v1"
SOFTMAX_REPLAY = ROOT / "evidence/verification/rtl-attention-score-token0-reference-batch-replay-v1"
HEAD0 = ROOT / "evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2"
PPA_SUMS = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/SHA256SUMS"
MERGE = ROOT / "evidence/diagnostics/c4-layer0-attention-value-offline-authority-inputs-merged-20260803-v1"
REPLAY = ROOT / "evidence/verification/rtl-attention-value-offline-v-authority-replay-v1"

EXPECTED_MISSION = "2c54f0f4291a894fa98a33ec0e7e82d50bf469cba160e9f80f2f9c288622ef43"
EXPECTED_REVIEW = "bfabcf84c1b63a90de800bfe49264a3f1ef191f9f60d7e872484e2a92b49ded7"
EXPECTED_OFFLINE_SUMS = "18e48c9cbe96345beac9c56784249d90e94f237f74e0bdc1c43bf5f09b371870"
EXPECTED_RECONSTRUCTION = "8e0a3d43579d636bb607e94f252b05eefa1cb8286b0b0119567d53244c84f3bf"
EXPECTED_ZERO_EXECUTION = "3a56b04f4807807b8af6f217ba347d086268dc731ce20c0c982a5e45c05b607b"
EXPECTED_INDEPENDENT_CHECK = "c745ebe1cda9dcf02d927025442711eb1ab1ce02f877266ebfbdfc9aa4d55d91"
EXPECTED_SOFTMAX_SUMS = "22a3addf99fd8afe757e62c303add0dbc5a97726a675550d8870aafcaf298604"
EXPECTED_SOFTMAX_REPORT = "1c7fda411e29449b1596e38e0d1cf44f94b38c220237e1e16db48cb26f38d115"
EXPECTED_SOFTMAX_BOUNDARY = "e39ab52fb9dd215d050919b0d07f261a6075d6cfec3caf26e8344e5a0e9c612f"
EXPECTED_HEAD0_SUMS = "7e3c768d9eff34b327859621e0e093243cf7562837611ad0bd37480a86703e97"
EXPECTED_HEAD0_WITNESS = "835c9d8f4432b2005998bac74c5536a9acd171f8042c33ac537af37d18869c3d"
EXPECTED_PPA_SUMS = "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4"
EXPECTED_REFERENCE = "c6260ab46e54fc0045253fb564566c9e1852a8806e477f7e9b9aaf6278c74b2e"
EXPECTED_SHELL_TB = "cefd1e4533f9e6de9f85583c0095f3eb2e81329c190f0e0044d045e31da76d18"
EXPECTED_SHELL = "3bb8caab4f06e6be9b170b5b3d91cb89b237715132e52507cd60f0514c61ab30"
EXPECTED_RTL_TREE = "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be"
EXPECTED_TOKEN0_V = "ae7b9d41e1b6fe625b111dc2b118f5007c821df23c036d927c25ca9e40fca565"
EXPECTED_SOFTMAX = "fdda20da03213e1cb0c63b7575d4397948b55f5bda4d096310266972b5d6fa54"
EXPECTED_TOKEN1_V = "341181d2a9c28549cac85a6c9b4532283b7e29d82e4011d8bef92f826f19a247"
EXPECTED_ATTENTION_VALUE = "a4bed51f31c0179ed0ba91db3bb75ef2435037041bb6bf773111598b8a5c8c48"

MISSION_ID = "rtl-attention-value-offline-v-authority-replay-v1"
DATASET = "c4_en_512"
RECORD = 64
LAYER = 0
QUERY_TOKEN = 1
HEAD = 0
HEAD_DIM = 64
HIDDEN = 896
CONTEXT = 2
ORDER = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
    "layer_0.rope_q",
    "layer_0.rope_k",
    "layer_0.kv_write",
    "layer_0.attention_score",
    "layer_0.softmax",
    "layer_0.attention_value",
    "layer_0.o_proj",
    "layer_0.attention_residual_add",
    "layer_0.post_attention_rmsnorm",
    "layer_0.mlp_gate_proj",
    "layer_0.mlp_up_proj",
    "layer_0.silu_gate",
    "layer_0.mlp_down_proj",
    "layer_0.mlp_residual_add",
]
PROTECTED_PATHS = [
    "design/RTL_MANIFEST.json",
    "design/RTL_MANIFEST.sha256",
    "design/PPA_FRONTIER_LEDGER.json",
    "design/RTL_TRACEABILITY.md",
    "design/RTL_TRACEABILITY.sha256",
    "research/PIPELINE_STATE.json",
    "research/PUBLIC_STATUS.json",
    "research/ENVIRONMENT_AUDIT.json",
]
FORBIDDEN_IMPORT_ROOTS = {"datasets", "safetensors", "tokenizers", "torch", "transformers"}
FORBIDDEN_CALL_NAMES = {
    "__call__",
    "calibrate",
    "capture",
    "forward",
    "from_pretrained",
    "replace_fixed_operators",
    "replace_linears",
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path, public_path: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": public_path or relative(path),
        "sha256": sha256_file(path),
    }


def validate_manifest(path: Path, base_dir: Path) -> int:
    count = 0
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row:
            continue
        expected, name = row.split("  ", 1)
        member = Path(name)
        member = member if member.is_absolute() else base_dir / member
        require(member.is_file(), f"missing manifest member: {name}")
        require(sha256_file(member) == expected, f"changed manifest member: {name}")
        count += 1
    return count


def write_sha256s(base_dir: Path) -> Path:
    output = base_dir / "SHA256SUMS"
    rows = []
    for member in sorted(base_dir.rglob("*")):
        if member.is_file() and member != output:
            rows.append(f"{sha256_file(member)}  {member.relative_to(base_dir).as_posix()}\n")
    output.write_text("".join(rows), encoding="utf-8")
    return output


def protected_state() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in PROTECTED_PATHS}


def rtl_tree() -> dict[str, Any]:
    rows = [(sha256_file(path), relative(path)) for path in sorted(ROOT.glob("rtl/**/*.sv"))]
    payload = "".join(f"{digest}  {name}\n" for digest, name in rows).encode()
    digest = sha256_bytes(payload)
    require(len(rows) == 23, f"RTL file count changed: {len(rows)}")
    require(digest == EXPECTED_RTL_TREE, f"RTL tree changed: {digest}")
    return {
        "file_count": len(rows),
        "sha256": digest,
        "hash_method": "path-sorted sha256 manifest over rtl/**/*.sv",
        "source_hashes": [{"path": name, "sha256": value} for value, name in rows],
    }


def tensor_raw(record: dict[str, Any]) -> bytes:
    path = ROOT / record["path"]
    raw = path.read_bytes()
    require(len(raw) == record["bytes"], f"tensor byte count changed: {path}")
    require(sha256_bytes(raw) == record["sha256"], f"tensor hash changed: {path}")
    return raw


def tensor_i8(record: dict[str, Any]) -> list[int]:
    return [value - 256 if value >= 128 else value for value in tensor_raw(record)]


def tensor_u16(record: dict[str, Any]) -> list[int]:
    raw = tensor_raw(record)
    require(len(raw) % 2 == 0, "uint16 tensor byte count is odd")
    return [int.from_bytes(raw[index : index + 2], "little") for index in range(0, len(raw), 2)]


def raw_i8(values: list[int]) -> bytes:
    return bytes(value & 0xFF for value in values)


def raw_i32(values: list[int]) -> bytes:
    return b"".join((value & 0xFFFFFFFF).to_bytes(4, "little") for value in values)


def pack(values: list[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (value & mask) << (lane * width)
    return packed


def sv_hex(value: int, width: int) -> str:
    return f"{width}'h{value & ((1 << width) - 1):0{width // 4}x}"


def static_zero_model_guard(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_imports: list[str] = []
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    forbidden_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                forbidden_imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in FORBIDDEN_CALL_NAMES:
                forbidden_calls.append(name)
    require(not forbidden_imports, f"forbidden model-stack imports: {forbidden_imports}")
    require(not forbidden_calls, f"forbidden model/capture calls: {forbidden_calls}")
    return {
        "ast_parsed": True,
        "forbidden_import_count": 0,
        "forbidden_call_count": 0,
        "forbidden_import_roots": sorted(FORBIDDEN_IMPORT_ROOTS),
        "forbidden_call_names": sorted(FORBIDDEN_CALL_NAMES),
    }


def find_operator(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [record for record in report["attempted_operators"] if record["operator"] == name]
    require(len(matches) == 1, f"operator record changed: {name}")
    return matches[0]


def validate_sources() -> dict[str, Any]:
    source_guard = static_zero_model_guard(SELF)
    require(sha256_file(MISSION) == EXPECTED_MISSION, "mission contract changed")
    require(sha256_file(REVIEW) == EXPECTED_REVIEW, "review decision changed")
    require(sha256_file(OFFLINE / "SHA256SUMS") == EXPECTED_OFFLINE_SUMS, "offline authority package changed")
    require(sha256_file(OFFLINE / "reconstruction.json") == EXPECTED_RECONSTRUCTION, "offline reconstruction changed")
    require(sha256_file(OFFLINE / "zero_model_execution.json") == EXPECTED_ZERO_EXECUTION, "offline zero-execution record changed")
    require(sha256_file(OFFLINE / "independent_check.json") == EXPECTED_INDEPENDENT_CHECK, "offline independent check changed")
    require(sha256_file(SOFTMAX_REPLAY / "SHA256SUMS") == EXPECTED_SOFTMAX_SUMS, "accepted softmax package changed")
    require(sha256_file(SOFTMAX_REPLAY / "ordered_prefix_report.json") == EXPECTED_SOFTMAX_REPORT, "accepted softmax report changed")
    require(sha256_file(SOFTMAX_REPLAY / "boundary_probe.json") == EXPECTED_SOFTMAX_BOUNDARY, "accepted softmax boundary changed")
    require(sha256_file(HEAD0 / "SHA256SUMS") == EXPECTED_HEAD0_SUMS, "immutable head0 package changed")
    require(sha256_file(HEAD0 / "witness.json") == EXPECTED_HEAD0_WITNESS, "immutable head0 witness changed")
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SUMS, "canonical PPA aggregate changed")
    require(sha256_file(ROOT / "tools/ace2_attention_value_reference.py") == EXPECTED_REFERENCE, "attention-value reference changed")
    require(sha256_file(ROOT / "verification/tb/ace2_shell_tb.sv") == EXPECTED_SHELL_TB, "maintained shell harness changed")
    require(sha256_file(ROOT / "rtl/ace2_shell.sv") == EXPECTED_SHELL, "shell RTL changed")
    member_counts = {
        "offline_authority": validate_manifest(OFFLINE / "SHA256SUMS", OFFLINE),
        "accepted_softmax": validate_manifest(SOFTMAX_REPLAY / "SHA256SUMS", SOFTMAX_REPLAY),
        "immutable_head0": validate_manifest(HEAD0 / "SHA256SUMS", HEAD0),
        "canonical_ppa": validate_manifest(PPA_SUMS, ROOT),
    }
    mission = json.loads(MISSION.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    reconstruction = json.loads((OFFLINE / "reconstruction.json").read_text(encoding="utf-8"))
    zero = json.loads((OFFLINE / "zero_model_execution.json").read_text(encoding="utf-8"))
    independent = json.loads((OFFLINE / "independent_check.json").read_text(encoding="utf-8"))
    score_report = json.loads((SOFTMAX_REPLAY / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    score_boundary = json.loads((SOFTMAX_REPLAY / "boundary_probe.json").read_text(encoding="utf-8"))
    head0 = json.loads((HEAD0 / "witness.json").read_text(encoding="utf-8"))
    require(mission["mission_id"] == MISSION_ID, "mission ID changed")
    require(review["review"]["status"] == "continue", "review no longer authorizes continuation")
    require("Hash-bind the accepted token0 V" in review["review"]["next_action"], "review next action changed")
    require(reconstruction["mission_id"] == MISSION_ID, "offline authority mission changed")
    require(reconstruction["derived_tensor_manifest"]["v_cache_head0_token0_s8"]["sha256"] == EXPECTED_TOKEN0_V, "token0 V authority changed")
    require(reconstruction["prior_candidate"]["authority"] is False, "old candidate was promoted to authority")
    require(reconstruction["prior_candidate"]["byte_exact_match"] is True, "old candidate cross-check changed")
    require(independent["status"] == "PASS_INDEPENDENT_FORWARD_FREE_RECONSTRUCTION", "offline independent check is not passing")
    for record in (zero, independent):
        for name in (
            "calibration_forward_count",
            "capture_count",
            "extraction_worker_count",
            "hook_count",
            "model_call_count",
            "model_execution_count",
            "model_forward_count",
        ):
            require(record[name] == 0, f"offline zero-execution counter changed: {name}")
    require(score_report["maximal_passing_prefix"] == ORDER[:9], "accepted prefix through softmax changed")
    require(score_report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.attention_value", "accepted softmax boundary changed")
    softmax_operator = find_operator(score_report, "layer_0.softmax")
    require(softmax_operator["status"] == "PASS_EXACT", "accepted softmax is not exact")
    require(softmax_operator["reference_hashes"]["probabilities_active_q15"] == EXPECTED_SOFTMAX, "accepted softmax payload hash changed")
    require(score_boundary["operator"] == "layer_0.attention_value", "softmax boundary operator changed")
    require(score_boundary["available_inputs"]["softmax_head0_query_token1_active_q15"]["sha256"] == EXPECTED_SOFTMAX, "softmax boundary payload changed")
    require(head0["tensor_manifest"]["kv_write_v_head0_token1_s8"]["sha256"] == EXPECTED_TOKEN1_V, "immutable token1 V changed")
    require(rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed")
    return {
        "source_guard": source_guard,
        "member_counts": member_counts,
        "mission": mission,
        "review": review,
        "reconstruction": reconstruction,
        "zero": zero,
        "independent": independent,
        "score_report": score_report,
        "score_boundary": score_boundary,
        "head0": head0,
    }


def merge_inputs() -> None:
    require(not MERGE.exists(), f"merge output already exists: {MERGE}")
    packages = validate_sources()
    reconstruction = packages["reconstruction"]
    score_report = packages["score_report"]
    score_boundary = packages["score_boundary"]
    head0 = packages["head0"]
    token0_v = reconstruction["derived_tensor_manifest"]["v_cache_head0_token0_s8"]
    probabilities = score_boundary["available_inputs"]["softmax_head0_query_token1_active_q15"]
    token1_v = head0["tensor_manifest"]["kv_write_v_head0_token1_s8"]
    require(score_boundary["available_inputs"]["v_cache_head0_token1_s8"] == token1_v, "softmax boundary token1 V differs from immutable head0")
    expected = head0["tensor_manifest"]["attention_value_head0_token1_s8"]
    o_proj_input = head0["tensor_manifest"]["o_proj_input_token1_s8"]
    o_proj_output = head0["tensor_manifest"]["o_proj_output_token1_s8"]
    require(token0_v["sha256"] == EXPECTED_TOKEN0_V, "token0 V authority hash changed")
    require(probabilities["sha256"] == EXPECTED_SOFTMAX, "softmax hash changed")
    require(token1_v["sha256"] == EXPECTED_TOKEN1_V, "token1 V hash changed")
    require(expected["sha256"] == EXPECTED_ATTENTION_VALUE, "attention-value reference changed")
    probability_values = tensor_u16(probabilities)
    token0_values = tensor_i8(token0_v)
    token1_values = tensor_i8(token1_v)
    expected_values = tensor_i8(expected)
    o_proj_values = tensor_i8(o_proj_input)
    require(len(probability_values) == CONTEXT, "active probability count changed")
    require(len(token0_values) == HEAD_DIM and len(token1_values) == HEAD_DIM, "active V shape changed")
    require(sum(probability_values) == 32768, "softmax probability sum changed")
    reference = reference_attention_value(
        AttentionValueCase(
            "c4_record64_layer0_head0_query1_offline_v_authority",
            probability_values,
            [token0_values, token1_values],
        )
    )
    require(reference.outputs == expected_values, "independent attention-value reference differs")
    require(reference.outputs == o_proj_values[:HEAD_DIM], "attention-value output is not the o_proj head0 slice")
    component_records = {
        "softmax_head0_query_token1_active_q15": probabilities,
        "v_cache_head0_token0_s8": token0_v,
        "v_cache_head0_token1_s8": token1_v,
    }
    concatenated = tensor_raw(probabilities) + tensor_raw(token0_v) + tensor_raw(token1_v)
    binding = {
        "component_order": [
            "softmax_head0_query_token1_active_q15",
            "v_cache_head0_token0_s8",
            "v_cache_head0_token1_s8",
        ],
        "component_hashes": {name: record["sha256"] for name, record in component_records.items()},
        "canonical_component_record_sha256": sha256_bytes(json_bytes(component_records)),
        "concatenated_payload_bytes": len(concatenated),
        "concatenated_payload_sha256": sha256_bytes(concatenated),
    }
    MERGE.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SELF, MERGE / "merge_source_at_execution.py")
    write_json(
        MERGE / "merge_manifest.json",
        {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "classification": "HASH_BOUND_OFFLINE_TOKEN0_V_ACCEPTED_SOFTMAX_IMMUTABLE_TOKEN1_V",
            "completed_at_utc": utc_now(),
            "coordinate": {
                "dataset": DATASET,
                "dataset_record_index": RECORD,
                "layer": LAYER,
                "query_token": QUERY_TOKEN,
                "active_tokens": [0, 1],
                "query_head": HEAD,
                "mapped_kv_head": 0,
            },
            "authority": {
                "token0_v": "forward_free_raw_weight_reconstruction_fresh_reviewer_accepted",
                "softmax": "accepted_unchanged_rtl_score_softmax_replay",
                "token1_v": "immutable_head0_package",
                "old_two_forward_candidate_authoritative": False,
                "old_two_forward_candidate_byte_exact_cross_check_only": True,
            },
            "source_packages": {
                "mission": artifact(MISSION, "handoff:rtl-attention-value-offline-v-authority-replay-v1/mission.json"),
                "review_decision": artifact(REVIEW, "handoff:rtl-attention-value-offline-v-authority-replay-v1/round-0001.json"),
                "offline_token0_v_authority": artifact(OFFLINE / "SHA256SUMS"),
                "offline_reconstruction": artifact(OFFLINE / "reconstruction.json"),
                "offline_zero_model_execution": artifact(OFFLINE / "zero_model_execution.json"),
                "offline_independent_check": artifact(OFFLINE / "independent_check.json"),
                "accepted_softmax_replay": artifact(SOFTMAX_REPLAY / "SHA256SUMS"),
                "accepted_softmax_report": artifact(SOFTMAX_REPLAY / "ordered_prefix_report.json"),
                "immutable_head0": artifact(HEAD0 / "SHA256SUMS"),
                "canonical_ppa": artifact(PPA_SUMS),
            },
            "logical_interfaces": {
                **component_records,
                "attention_value_head0_token1_reference_s8": expected,
                "o_proj_input_token1_s8": o_proj_input,
                "o_proj_output_token1_s8": o_proj_output,
            },
            "hash_bound_input": binding,
            "reference": {
                "accumulator_s32_sha256": sha256_bytes(raw_i32(reference.accumulators)),
                "output_s8_sha256": sha256_bytes(raw_i8(reference.outputs)),
                "saturation_seen": reference.saturation_seen,
                "probability_sum_q15": sum(probability_values),
                "output_equals_retained_reference": True,
                "output_equals_o_proj_input_head0_slice": True,
            },
            "attention_value_metadata": {
                "probability_format": "unsigned_int16_Q0.15",
                "value_cache_format": "signed_int8_head_dim_64",
                "accumulator_format": "signed_int32",
                "rounding": "round_to_nearest_ties_to_even_after_15_fractional_bits",
                "output_format": "signed_int8",
                "v_projection_output_scale": reconstruction["fixed_point_contract"]["v_projection_output_scale"],
            },
            "accepted_prefix": {
                "maximal_passing_prefix": score_report["maximal_passing_prefix"],
                "rtl_tree_sha256": score_report["rtl_tree"]["sha256"],
            },
            "zero_model_execution": {
                "model_execution_count": 0,
                "model_forward_count": 0,
                "model_call_count": 0,
                "calibration_forward_count": 0,
                "capture_count": 0,
                "hook_count": 0,
                "extraction_worker_count": 0,
                "static_source_guard": packages["source_guard"],
            },
            "member_counts": packages["member_counts"],
            "source": artifact(MERGE / "merge_source_at_execution.py"),
            "protected_state_before": protected_state(),
            "source_packages_modified": False,
            "state_mutations": {
                "rtl": False,
                "frontier_authority_seal": False,
                "synthesis_opensta_ppa": False,
                "baseline_candidate_benchmark_scale32": False,
            },
        },
    )
    write_sha256s(MERGE)


def render_attention_value_vectors(
    probabilities: list[int], values: list[list[int]], expected: list[int], saturation: bool
) -> str:
    padded_probabilities = probabilities + [0] * (8 - len(probabilities))
    padded_values = values + [[0] * HEAD_DIM for _ in range(8 - len(values))]
    lines = [
        "// Generated from the hash-bound offline token0 V authority package.",
        "localparam integer ATTN_VALUE_CASE_COUNT = 1;",
        "localparam integer ATTN_VALUE_CONTEXT_MAX = 8;",
        "localparam integer ATTN_VALUE_HEAD_DIM = 64;",
        "localparam integer ATTN_VALUE_BEATS_PER_VECTOR = 4;",
        "reg [15:0] attn_value_context_count [0:ATTN_VALUE_CASE_COUNT-1];",
        "reg [16*8-1:0] attn_value_probability_word [0:ATTN_VALUE_CASE_COUNT-1];",
        "reg [127:0] attn_value_v_beats [0:ATTN_VALUE_CASE_COUNT*ATTN_VALUE_CONTEXT_MAX*ATTN_VALUE_BEATS_PER_VECTOR-1];",
        "reg [127:0] attn_value_expected_beats [0:ATTN_VALUE_CASE_COUNT*ATTN_VALUE_BEATS_PER_VECTOR-1];",
        "reg attn_value_expected_saturation [0:ATTN_VALUE_CASE_COUNT-1];",
        "initial begin",
        f"  attn_value_context_count[0] = 16'd{len(probabilities)};",
        f"  attn_value_probability_word[0] = {sv_hex(pack(padded_probabilities, 16), 128)};",
    ]
    for token, row in enumerate(padded_values):
        for beat in range(4):
            lines.append(
                f"  attn_value_v_beats[{token * 4 + beat}] = "
                f"{sv_hex(pack(row[beat * 16:(beat + 1) * 16], 8), 128)};"
            )
    for beat in range(4):
        lines.append(
            f"  attn_value_expected_beats[{beat}] = "
            f"{sv_hex(pack(expected[beat * 16:(beat + 1) * 16], 8), 128)};"
        )
    lines.append(f"  attn_value_expected_saturation[0] = 1'b{int(saturation)};")
    lines.append("end")
    return "\n".join(lines) + "\n"


def execution_testbench() -> str:
    text = (ROOT / "verification/tb/ace2_shell_tb.sv").read_text(encoding="utf-8")
    text = text.replace(
        '    `include "../generated/attention_value_vectors.svh"',
        '    `include "c4_attention_value_offline_authority_vectors.svh"',
        1,
    )
    require('`include "c4_attention_value_offline_authority_vectors.svh"' in text, "attention-value include replacement failed")
    text = text.replace(
        "    integer attn_score_only_mode;",
        "    integer attn_score_only_mode;\n    integer c4_attention_value_offline_authority_replay_mode;",
        1,
    )
    text = text.replace(
        '        attn_score_only_mode = $test$plusargs("ATTN_SCORE_ONLY");',
        '        attn_score_only_mode = $test$plusargs("ATTN_SCORE_ONLY");\n'
        '        c4_attention_value_offline_authority_replay_mode = $test$plusargs("C4_ATTENTION_VALUE_OFFLINE_AUTHORITY_REPLAY");',
        1,
    )
    branch = (
        "        if (c4_attention_value_offline_authority_replay_mode) begin\n"
        "            send_attn_value_cmd(0, 0, 16'h3e03);\n"
        "            wait_attn_value_done_and_compare(0, 0, 16'h3e03);\n"
        "            if (failures != 0) begin\n"
        "                $display(\"ACE2_C4_ATTENTION_VALUE_OFFLINE_AUTHORITY_FAIL failures=%0d\", failures);\n"
        "                $fatal(1, \"ACE2_C4_ATTENTION_VALUE_OFFLINE_AUTHORITY_FAIL\");\n"
        "            end\n"
        "            $display(\"ACE2_C4_ATTENTION_VALUE_OFFLINE_AUTHORITY_PASS writes=%0d cycles=%0d saturation=%0d out0=%032x out1=%032x out2=%032x out3=%032x\", observed_count, last_command_cycles, attn_value_expected_saturation[0], observed_output[0], observed_output[1], observed_output[2], observed_output[3]);\n"
        "            $display(\"ACE2_C4_ATTENTION_VALUE_OFFLINE_AUTHORITY_ORDERED_REPLAY_PASS operators=1\");\n"
        "            $finish;\n"
        "        end\n\n"
    )
    needle = "        if (qproj_stride_only_mode) begin"
    require(needle in text, "focused replay insertion point changed")
    return text.replace(needle, branch + needle, 1)


def run_replay() -> None:
    require(MERGE.is_dir(), "hash-bound merge package is missing")
    require(not REPLAY.exists(), f"replay output already exists: {REPLAY}")
    packages = validate_sources()
    validate_manifest(MERGE / "SHA256SUMS", MERGE)
    merge = json.loads((MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    require(merge["mission_id"] == MISSION_ID, "merge mission changed")
    require(merge["authority"]["old_two_forward_candidate_authoritative"] is False, "old candidate became authoritative")
    require(merge["zero_model_execution"]["model_execution_count"] == 0, "merge model execution count changed")
    logical = merge["logical_interfaces"]
    probabilities = tensor_u16(logical["softmax_head0_query_token1_active_q15"])
    token0_v = tensor_i8(logical["v_cache_head0_token0_s8"])
    token1_v = tensor_i8(logical["v_cache_head0_token1_s8"])
    expected = tensor_i8(logical["attention_value_head0_token1_reference_s8"])
    o_proj_input = tensor_i8(logical["o_proj_input_token1_s8"])
    reference = reference_attention_value(
        AttentionValueCase(
            "c4_record64_layer0_head0_query1_offline_v_authority",
            probabilities,
            [token0_v, token1_v],
        )
    )
    require(reference.outputs == expected, "attention-value independent reference differs")
    require(reference.outputs == o_proj_input[:HEAD_DIM], "attention-value/o_proj continuity changed")
    require(sum(probabilities) == 32768, "probability normalization changed")
    require(shutil.which("iverilog") is not None, "iverilog is unavailable")
    require(shutil.which("vvp") is not None, "vvp is unavailable")
    REPLAY.mkdir(parents=True, exist_ok=False)
    generated = REPLAY / "generated"
    generated.mkdir()
    vectors = generated / "c4_attention_value_offline_authority_vectors.svh"
    vectors.write_text(
        render_attention_value_vectors(probabilities, [token0_v, token1_v], reference.outputs, reference.saturation_seen),
        encoding="utf-8",
    )
    shutil.copy2(SELF, REPLAY / "replay_source_at_execution.py")
    tb = REPLAY / "replay_tb_source_at_execution.sv"
    tb.write_text(execution_testbench(), encoding="utf-8")
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    image = ROOT / "build/rtl-attention-value-offline-v-authority-replay-v1/ordered.vvp"
    image.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [
        "iverilog",
        "-g2012",
        "-Wall",
        "-Irtl",
        "-Irtl/generated",
        "-Iverification/generated",
        "-Iverification/tb",
        f"-I{generated}",
        "-s",
        "ace2_shell_tb",
        "-o",
        str(image),
        *[str(path) for path in rtl_sources],
        str(tb),
    ]
    compiled = subprocess.run(compile_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "iverilog.log").write_text(
        "command=" + " ".join(compile_command) + "\n" + compiled.stdout + compiled.stderr,
        encoding="utf-8",
    )
    require(compiled.returncode == 0, "unchanged-tree attention-value compilation failed")
    replay_command = ["vvp", str(image), "+C4_ATTENTION_VALUE_OFFLINE_AUTHORITY_REPLAY"]
    replayed = subprocess.run(replay_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "replay.stdout").write_text(replayed.stdout, encoding="utf-8")
    (REPLAY / "replay.stderr").write_text(replayed.stderr, encoding="utf-8")
    (REPLAY / "replay.log").write_text(
        "command=" + " ".join(replay_command) + "\n" + replayed.stdout + replayed.stderr,
        encoding="utf-8",
    )
    require(replayed.returncode == 0, "unchanged-tree attention-value execution failed")
    match = re.search(
        r"ACE2_C4_ATTENTION_VALUE_OFFLINE_AUTHORITY_PASS writes=(\d+) cycles=(\d+) saturation=(\d+) "
        r"out0=([0-9a-fA-F]+) out1=([0-9a-fA-F]+) out2=([0-9a-fA-F]+) out3=([0-9a-fA-F]+)",
        replayed.stdout,
    )
    require(match is not None, "attention-value replay pass marker missing")
    require(
        "ACE2_C4_ATTENTION_VALUE_OFFLINE_AUTHORITY_ORDERED_REPLAY_PASS operators=1" in replayed.stdout,
        "ordered replay completion marker missing",
    )
    require("MISMATCH" not in replayed.stdout and "_FAIL" not in replayed.stdout, "ordered replay mismatch reported")
    require(int(match.group(1)) == 4, "attention-value write count changed")
    require(bool(int(match.group(3))) == reference.saturation_seen, "attention-value saturation result changed")
    rtl_raw = b"".join(int(match.group(index), 16).to_bytes(16, "little") for index in range(4, 8))
    require(rtl_raw == raw_i8(expected), "RTL attention-value bytes differ from retained reference")
    boundary = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "operator": "layer_0.o_proj",
        "classification": "UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_RETAINED_O_PROJ_WEIGHTS_METADATA_INCOMPLETE",
        "available_inputs": {
            "o_proj_input_token1_s8": logical["o_proj_input_token1_s8"],
            "o_proj_output_token1_reference_s8": logical["o_proj_output_token1_s8"],
            "head0_input_slice_from_attention_value": logical["attention_value_head0_token1_reference_s8"],
        },
        "continuity": {
            "attention_value_output_equals_o_proj_input_head0_slice": True,
            "compared_bytes": HEAD_DIM,
            "full_o_proj_input_bytes": HIDDEN,
        },
        "missing_fields": [
            "layer_0.o_proj.input.weights_s4[896][896]",
            "layer_0.o_proj.input.bias_accumulator_s32[896]",
            "layer_0.o_proj.input.multiplier_s32[896]",
            "layer_0.o_proj.input.right_shift_u6[896]",
        ],
        "scope_reason": "The retained immutable package has exact o_proj input/output tensors but not the exact weights and per-channel requantization metadata. This bounded mission forbids new capture and stops at this existing interface boundary.",
        "simulation_executed": False,
        "later_operators_attempted": False,
    }
    write_json(REPLAY / "boundary_probe.json", boundary)
    zero_execution = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "completed_at_utc": utc_now(),
        "status": "PASS_ZERO_MODEL_EXECUTION_REPLAY_ONLY_SUBPROCESSES",
        "model_execution_count": 0,
        "model_forward_count": 0,
        "model_call_count": 0,
        "calibration_forward_count": 0,
        "capture_count": 0,
        "hook_count": 0,
        "extraction_worker_count": 0,
        "subprocess_count": 2,
        "subprocess_classes": ["iverilog_compile", "vvp_simulation"],
        "static_source_guard": packages["source_guard"],
        "source": artifact(REPLAY / "replay_source_at_execution.py"),
    }
    write_json(REPLAY / "zero_model_execution.json", zero_execution)
    protected_after = protected_state()
    ppa_hash_after = sha256_file(PPA_SUMS)
    rtl_after = rtl_tree()
    require(protected_after == merge["protected_state_before"], "protected state changed during replay")
    require(ppa_hash_after == EXPECTED_PPA_SUMS, "canonical PPA manifest changed during replay")
    require(rtl_after["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed during replay")
    attention_value_record = {
        "operator": "layer_0.attention_value",
        "status": "PASS_EXACT",
        "input_hashes": {
            "probabilities_active_q15": logical["softmax_head0_query_token1_active_q15"]["sha256"],
            "v_token0_head0_offline_authority": logical["v_cache_head0_token0_s8"]["sha256"],
            "v_token1_head0_immutable": logical["v_cache_head0_token1_s8"]["sha256"],
            "hash_bound_input_payload": merge["hash_bound_input"]["concatenated_payload_sha256"],
            "hash_bound_component_records": merge["hash_bound_input"]["canonical_component_record_sha256"],
        },
        "reference_hashes": {
            "accumulator_s32": sha256_bytes(raw_i32(reference.accumulators)),
            "attention_value_head0_s8": logical["attention_value_head0_token1_reference_s8"]["sha256"],
        },
        "output_hashes": {
            "rtl_attention_value_head0_s8": sha256_bytes(rtl_raw),
        },
        "rtl_source_hashes": {
            "rtl_tree": rtl_after["sha256"],
            "shell": artifact(ROOT / "rtl/ace2_shell.sv"),
            "maintained_harness": artifact(ROOT / "verification/tb/ace2_shell_tb.sv"),
            "execution_harness": artifact(tb),
            "generated_vectors": artifact(vectors),
        },
        "counts": {
            "active_tokens": CONTEXT,
            "probability_elements": CONTEXT,
            "value_input_elements": CONTEXT * HEAD_DIM,
            "output_elements": HEAD_DIM,
            "compared_outputs": HEAD_DIM,
            "mismatches": 0,
            "rtl_output_writes": int(match.group(1)),
        },
        "cycles": {
            "total_command_cycles": int(match.group(2)),
        },
        "handshake_latency": {
            "simulation_executed": True,
            "handshake_result": "PASS_NO_TIMEOUT_OR_DROPPED_WRITE",
            "latency_result": "PASS_BOUNDED_BY_MAINTAINED_SHELL_WATCHDOG",
        },
        "invariants": {
            "probability_sum_q15": sum(probabilities),
            "probability_sum_expected_q15": 32768,
            "reference_saturation_seen": reference.saturation_seen,
            "rtl_saturation_matches_reference": True,
            "output_byte_exact_reference_match": True,
            "output_equals_o_proj_input_head0_slice": True,
            "rtl_tree_unchanged": True,
            "model_execution_count": 0,
            "capture_count": 0,
        },
        "raw_log": artifact(REPLAY / "replay.log"),
    }
    report = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "producer_role": "engineer",
        "review_status": "PENDING_FRESH_REVIEWER",
        "completed_at_utc": utc_now(),
        "coordinate": {
            "dataset": DATASET,
            "dataset_record_index": RECORD,
            "layer": LAYER,
            "query_token": QUERY_TOKEN,
            "active_tokens": [0, 1],
            "query_head": HEAD,
            "mapped_kv_head": 0,
        },
        "authority_binding": {
            "merge_manifest": artifact(MERGE / "merge_manifest.json"),
            "hash_bound_input": merge["hash_bound_input"],
            "token0_v_authority": "forward_free_raw_weight_reconstruction",
            "old_two_forward_candidate_authority": False,
            "old_two_forward_candidate_used_for_replay": False,
        },
        "zero_model_execution": artifact(REPLAY / "zero_model_execution.json"),
        "continuous_replay": {
            "start_operator": "layer_0.attention_value",
            "last_passing_operator": "layer_0.attention_value",
            "first_real_boundary": "layer_0.o_proj",
            "later_operators_attempted": False,
        },
        "frozen_order": ORDER,
        "rtl_tree": rtl_after,
        "prior_maximal_prefix": ORDER[:9],
        "attempted_operators": [
            attention_value_record,
            {
                "operator": "layer_0.o_proj",
                "status": boundary["classification"],
                "input_hashes": {
                    "o_proj_input_token1_s8": logical["o_proj_input_token1_s8"]["sha256"],
                    "o_proj_output_reference_token1_s8": logical["o_proj_output_token1_s8"]["sha256"],
                    "attention_value_head0_slice_s8": logical["attention_value_head0_token1_reference_s8"]["sha256"],
                },
                "counts": {
                    "retained_input_bytes": HIDDEN,
                    "retained_output_reference_bytes": HIDDEN,
                    "missing_weight_elements": HIDDEN * HIDDEN,
                    "missing_metadata_channels": HIDDEN,
                    "mismatch_count": None,
                },
                "cycles": {"total_command_cycles": None},
                "handshake_latency": {
                    "simulation_executed": False,
                    "handshake_result": "NOT_EVALUATED_INTERFACE_UNAVAILABLE",
                    "latency_result": "NOT_EVALUATED_INTERFACE_UNAVAILABLE",
                },
                "invariants": {
                    "attention_value_output_equals_o_proj_input_head0_slice": True,
                    "later_operators_attempted": False,
                },
                "boundary_probe": artifact(REPLAY / "boundary_probe.json"),
            },
        ],
        "newly_passing_operators": ["layer_0.attention_value"],
        "maximal_passing_prefix": ORDER[:10],
        "first_failing_or_unsupported_operator": {
            "operator": "layer_0.o_proj",
            "classification": boundary["classification"],
            "boundary_probe": artifact(REPLAY / "boundary_probe.json"),
        },
        "batch_result": "STOPPED_AT_FIRST_PRECISELY_UNAVAILABLE_EXECUTABLE_INTERFACE",
        "later_operators_attempted": False,
        "protected_state_pre_post_exact_match": True,
        "canonical_ppa": {
            "reused_not_rerun": True,
            "sha256_manifest": artifact(PPA_SUMS),
            "pre_post_exact_match": True,
        },
        "forbidden_flows_run": [],
        "state_mutations": {
            "rtl": False,
            "frontier_ledger_manifest_traceability_pipeline_seals": False,
            "synthesis_opensta_ppa": False,
            "baseline_candidate_benchmark_scale32": False,
        },
        "review_requirement": "Fresh Reviewer must independently accept or reject this offline-authority maximal-prefix result.",
    }
    write_json(REPLAY / "ordered_prefix_report.json", report)
    (REPLAY / "batch_replay.log").write_text(
        "ACE2_ATTENTION_VALUE_OFFLINE_V_AUTHORITY_ORDERED_BATCH_REPLAY\n"
        f"rtl_tree_sha256={EXPECTED_RTL_TREE}\n"
        f"hash_bound_input_sha256={merge['hash_bound_input']['concatenated_payload_sha256']}\n"
        f"token0_v_sha256={EXPECTED_TOKEN0_V}\n"
        f"softmax_sha256={EXPECTED_SOFTMAX}\n"
        f"token1_v_sha256={EXPECTED_TOKEN1_V}\n"
        "model_execution_count=0\n"
        "capture_count=0\n"
        f"layer_0.attention_value=PASS_EXACT outputs=64 mismatches=0 cycles={int(match.group(2))}\n"
        "layer_0.o_proj=UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_RETAINED_O_PROJ_WEIGHTS_METADATA_INCOMPLETE\n"
        "maximal_passing_operator=layer_0.attention_value\n"
        "later_operators_attempted=false\n"
        "synthesis_opensta_ppa_executed=false\n",
        encoding="utf-8",
    )
    write_sha256s(REPLAY)
    print(
        "ACE2_ATTENTION_VALUE_OFFLINE_V_AUTHORITY_REPLAY_BOUNDARY "
        f"input_sha256={merge['hash_bound_input']['concatenated_payload_sha256']} "
        f"cycles={int(match.group(2))} maximal_prefix=layer_0.attention_value "
        "first_boundary=layer_0.o_proj mismatches=0 model_execution_count=0"
    )


def run_check() -> None:
    validate_sources()
    merge_members = validate_manifest(MERGE / "SHA256SUMS", MERGE)
    replay_members = validate_manifest(REPLAY / "SHA256SUMS", REPLAY)
    merge = json.loads((MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((REPLAY / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    zero = json.loads((REPLAY / "zero_model_execution.json").read_text(encoding="utf-8"))
    require(sha256_file(REPLAY / "replay_source_at_execution.py") == sha256_file(SELF), "archived replay source changed")
    require(merge["logical_interfaces"]["v_cache_head0_token0_s8"]["sha256"] == EXPECTED_TOKEN0_V, "merged token0 V changed")
    require(merge["logical_interfaces"]["softmax_head0_query_token1_active_q15"]["sha256"] == EXPECTED_SOFTMAX, "merged softmax changed")
    require(merge["logical_interfaces"]["v_cache_head0_token1_s8"]["sha256"] == EXPECTED_TOKEN1_V, "merged token1 V changed")
    require(report["newly_passing_operators"] == ["layer_0.attention_value"], "passing operator set changed")
    require(report["maximal_passing_prefix"] == ORDER[:10], "maximal prefix changed")
    require(report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.o_proj", "first boundary changed")
    require(report["later_operators_attempted"] is False, "replay continued beyond first boundary")
    attention = find_operator(report, "layer_0.attention_value")
    require(attention["status"] == "PASS_EXACT", "attention value is not exact")
    require(attention["counts"]["mismatches"] == 0, "attention value mismatch count changed")
    require(attention["output_hashes"]["rtl_attention_value_head0_s8"] == EXPECTED_ATTENTION_VALUE, "RTL output hash changed")
    require(attention["invariants"]["probability_sum_q15"] == 32768, "probability invariant changed")
    require(attention["invariants"]["output_equals_o_proj_input_head0_slice"] is True, "o_proj continuity changed")
    for name in (
        "model_execution_count",
        "model_forward_count",
        "model_call_count",
        "calibration_forward_count",
        "capture_count",
        "hook_count",
        "extraction_worker_count",
    ):
        require(zero[name] == 0, f"replay zero-execution counter changed: {name}")
    require(zero["subprocess_classes"] == ["iverilog_compile", "vvp_simulation"], "replay subprocess scope changed")
    require(protected_state() == merge["protected_state_before"], "protected state changed")
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SUMS, "canonical PPA manifest changed")
    require(rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed")
    print(
        "ACE2_ATTENTION_VALUE_OFFLINE_V_AUTHORITY_REPLAY_CHECK_PASS "
        f"merge_members={merge_members} replay_members={replay_members} "
        f"input_sha256={merge['hash_bound_input']['concatenated_payload_sha256']} "
        f"report_sha256={sha256_file(REPLAY / 'ordered_prefix_report.json')} "
        "mismatches=0 model_execution_count=0 first_boundary=layer_0.o_proj"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "check"))
    args = parser.parse_args()
    if args.mode == "run":
        merge_inputs()
        run_replay()
    else:
        run_check()


if __name__ == "__main__":
    main()
