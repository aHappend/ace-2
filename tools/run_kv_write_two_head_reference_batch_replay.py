#!/usr/bin/env python3
"""Capture only KV head 1, merge with immutable head 0, and replay KV_WRITE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_layer0_minimal_reference_capture_batch_replay as base


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-kv-write-two-head-reference-batch-replay-v1/mission.json"
)
HEAD0 = ROOT / "evidence/diagnostics/c4-layer0-minimal-reference-capture-20260803-v2"
HEAD0_SUMS = HEAD0 / "SHA256SUMS"
CAPTURE = ROOT / "evidence/diagnostics/c4-layer0-kv-head1-minimal-reference-capture-20260803-v1"
MERGE = ROOT / "evidence/diagnostics/c4-layer0-kv-write-two-head-merged-20260803-v1"
REPLAY = ROOT / "evidence/verification/rtl-kv-write-two-head-reference-batch-replay-v1"
PPA_SUMS = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/SHA256SUMS"

EXPECTED_RTL_TREE = "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be"
EXPECTED_HEAD0_SUMS = "7e3c768d9eff34b327859621e0e093243cf7562837611ad0bd37480a86703e97"
EXPECTED_HEAD0_SOURCE = "c23fb184455f45e5f60250e7cab80f5cb4f152f86c5e4270bd75082e0252b1ef"
EXPECTED_SHELL_TB = "cefd1e4533f9e6de9f85583c0095f3eb2e81329c190f0e0044d045e31da76d18"
EXPECTED_PPA_SUMS = "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4"
TOKEN = 1
HEAD = 1
HEAD_DIM = 64
ORDER = base.ORDER


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path, public_path: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": public_path or relative(path),
        "sha256": sha256_file(path),
    }


def external_artifact(path: Path, public_path: str) -> dict[str, Any]:
    require(path.is_file(), f"missing external artifact: {path}")
    return {"bytes": path.stat().st_size, "path": public_path, "sha256": sha256_file(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def head0_witness() -> dict[str, Any]:
    require(sha256_file(HEAD0_SUMS) == EXPECTED_HEAD0_SUMS, "immutable head-0 aggregate changed")
    validate_manifest(HEAD0_SUMS, HEAD0)
    witness = json.loads((HEAD0 / "witness.json").read_text(encoding="utf-8"))
    require(len(witness["tensor_manifest"]) == 30, "head-0 tensor count changed")
    require(witness["capture"]["evaluation_forward_count"] == 1, "head-0 forward count changed")
    require(witness["capture"]["head_count_retained"] == 1, "head-0 retained-head count changed")
    require(witness["coordinate"]["head"] == 0, "head-0 coordinate changed")
    require(witness["coordinate"]["dataset_record_index"] == 64, "head-0 record changed")
    require(witness["coordinate"]["layer"] == 0 and witness["coordinate"]["token"] == TOKEN, "head-0 layer/token changed")
    require(witness["source_bindings"]["capture_tool"]["sha256"] == EXPECTED_HEAD0_SOURCE, "head-0 capture source changed")
    return witness


def preflight_payload() -> dict[str, Any]:
    witness = head0_witness()
    require(base.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed")
    require(sha256_file(ROOT / "verification/tb/ace2_shell_tb.sv") == EXPECTED_SHELL_TB, "maintained shell testbench changed")
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SUMS, "canonical PPA aggregate changed")
    live_binding, _calibration, _evaluation, manifest, versions = base.live_input_binding()
    require(live_binding == witness["input_binding"], "live input binding differs from head 0")
    cached = base.cached_model_bindings(manifest["model"]["repository"], manifest["model"]["revision"])
    head0_preflight = json.loads((HEAD0 / "preflight.json").read_text(encoding="utf-8"))
    require(cached == head0_preflight["model"]["cached_files"], "pinned model files differ from head 0")
    for binding in witness["source_bindings"].values():
        name = binding.get("path")
        if name and name.startswith(("tools/", "benchmark/")):
            require(sha256_file(ROOT / name) == binding["sha256"], f"pinned source changed: {name}")
    tools = {name: shutil.which(name) for name in ("iverilog", "vvp")}
    require(all(tools.values()), f"required RTL tools unavailable: {tools}")
    return {
        "schema_version": 1,
        "mission_id": "rtl-kv-write-two-head-reference-batch-replay-v1",
        "status": "PASS_MODEL_EXECUTION_NOT_CONSUMED",
        "completed_at_utc": utc_now(),
        "model_execution_count": 0,
        "coordinate": {
            "dataset": "c4_en_512",
            "dataset_record_index": 64,
            "head": HEAD,
            "layer": 0,
            "token": TOKEN,
        },
        "input_binding": live_binding,
        "head0_compatibility": {
            "aggregate": artifact(HEAD0_SUMS),
            "tensor_count": 30,
            "witness": artifact(HEAD0 / "witness.json"),
        },
        "mission": external_artifact(MISSION, "handoff:rtl-kv-write-two-head-reference-batch-replay-v1/mission.json"),
        "model": {
            "repository": manifest["model"]["repository"],
            "revision": manifest["model"]["revision"],
            "cached_files": cached,
        },
        "rtl_tree": base.rtl_tree(),
        "runtime": {
            "packages": versions,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "source": artifact(SELF),
        "tool_lookup": tools,
    }


def run_preflight() -> None:
    require(not (CAPTURE / "capture_execution_consumed.json").exists(), "capture authorization already consumed")
    require(not (CAPTURE / "witness.json").exists(), "capture witness already exists")
    CAPTURE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SELF, CAPTURE / "preflight_source_at_execution.py")
    payload = preflight_payload()
    payload["source_archive"] = artifact(CAPTURE / "preflight_source_at_execution.py")
    write_json(CAPTURE / "preflight.json", payload)
    (CAPTURE / "preflight.log").write_text(
        "ACE2_KV_HEAD1_PREFLIGHT_PASS\n"
        "model_execution_count=0\n"
        "coordinate=c4_en_512:record64:layer0:token1:head1\n"
        f"rtl_tree_sha256={EXPECTED_RTL_TREE}\n"
        f"head0_sha256s_sha256={EXPECTED_HEAD0_SUMS}\n",
        encoding="utf-8",
    )
    print("ACE2_KV_HEAD1_PREFLIGHT_PASS model_execution_count=0")


def validate_preflight() -> dict[str, Any]:
    path = CAPTURE / "preflight.json"
    require(path.is_file(), "preflight artifact missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["status"] == "PASS_MODEL_EXECUTION_NOT_CONSUMED", "preflight status changed")
    require(payload["model_execution_count"] == 0, "preflight consumed execution")
    require(artifact(CAPTURE / "preflight_source_at_execution.py") == payload["source_archive"], "preflight source archive changed")
    if (CAPTURE / "capture_execution_consumed.json").exists():
        capture_source = CAPTURE / "capture_source_at_execution.py"
        require(capture_source.is_file(), "consumed capture source archive missing")
        require(sha256_file(capture_source) == payload["source"]["sha256"], "consumed capture source changed")
    else:
        require(sha256_file(SELF) == payload["source"]["sha256"], "capture tool changed after preflight")
    require(base.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL changed after preflight")
    require(sha256_file(HEAD0_SUMS) == EXPECTED_HEAD0_SUMS, "head-0 package changed after preflight")
    return payload


def scale32_record(value: float) -> int:
    record = int(base.fixed.ceil_scale32_from_float(float(value)))
    require(0 <= record <= 0xFFFFFFFF, f"invalid Scale32 record: {record}")
    return record


def execute_capture_worker() -> None:
    preflight = validate_preflight()
    marker = CAPTURE / "capture_execution_consumed.json"
    source_archive = CAPTURE / "capture_source_at_execution.py"
    require(marker.is_file(), "consumption marker missing")
    require(source_archive.is_file(), "capture source archive missing")
    require(sha256_file(source_archive) == preflight["source"]["sha256"], "capture source differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("capture requires CUDA_VISIBLE_DEVICES empty or -1")

    live_binding, calibration_prompt, evaluation_prompt, manifest, versions = base.live_input_binding()
    require(live_binding == preflight["input_binding"], "live input changed after preflight")
    config = base.load_contracts(require_rtl_binding=False)[1]
    base.seed_everything(config)
    model_spec = manifest["model"]
    model = base.AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "model is not CPU-only")
    require(getattr(model.config, "_commit_hash", None) == base.EXPECTED_REVISION, "resolved revision changed")
    ranges, operator_ranges = base.calibrate(model, [calibration_prompt])
    base.replace_linears(model, ranges, rope_diagnostic_mechanism=None)
    base.replace_fixed_operators(model, operator_ranges, rope_diagnostic_mechanism=None)
    layer = model.model.layers[0]
    require(isinstance(layer, base.FixedDecoderLayer), "layer 0 is not FixedDecoderLayer")
    require(isinstance(layer.self_attn, base.FixedAttention), "layer 0 attention is not fixed")
    trace = base.ExtractionTrace(layer)
    trace.install()
    base.seed_everything(config)
    evaluation_forward_count = 0
    try:
        with torch.inference_mode():
            evaluation_forward_count += 1
            model(input_ids=evaluation_prompt, use_cache=False)
    except base.Layer0CaptureComplete:
        pass
    finally:
        trace.restore()
    require(evaluation_forward_count == 1, "retained-coordinate forward count changed")
    require(len(trace.calls.get("fixed_rope_raw_with_saturation", [])) == 2, "expected Q and K RoPE calls")

    k_rope_all, k_saturations = trace.calls["fixed_rope_raw_with_saturation"][1]["result"]
    k_head1 = k_rope_all[0, HEAD, TOKEN].to(torch.int8)
    v_all = trace.methods["v_proj"]["output"]
    v_head1 = v_all[0, TOKEN, HEAD * HEAD_DIM : (HEAD + 1) * HEAD_DIM].to(torch.int8)
    require(k_head1.numel() == HEAD_DIM and v_head1.numel() == HEAD_DIM, "head-1 shape changed")

    records = [
        scale32_record(float(layer.self_attn.key_rope_output_scales[0])),
        scale32_record(float(layer.self_attn.key_rope_output_scales[1])),
        scale32_record(float(layer.self_attn.v_proj.output_scale)),
        scale32_record(float(layer.self_attn.v_proj.output_scale)),
    ]
    metadata = torch.tensor(list(struct.pack("<IIII", *records)), dtype=torch.uint8)
    tensors = {
        "kv_write_k_head1_token1_s8": base.write_tensor(CAPTURE, "kv_write_k_head1_token1_s8", k_head1),
        "kv_write_v_head1_token1_s8": base.write_tensor(CAPTURE, "kv_write_v_head1_token1_s8", v_head1),
        "kv_write_joint_two_head_scale_metadata_u8": base.write_tensor(
            CAPTURE, "kv_write_joint_two_head_scale_metadata_u8", metadata
        ),
    }

    h0 = head0_witness()
    require(records[0] == scale32_record(h0["rope_k"]["output_scale"]), "head-0 K scale compatibility failed")
    require(records[2] == scale32_record(h0["operator_metadata"]["v_projection_output_scale"]), "head-0 V scale compatibility failed")
    witness = {
        "schema_version": 1,
        "classification": "single_retained_coordinate_extraction_only_kv_head1_and_joint_metadata",
        "capture": {
            "calibration_dependency_forward_count": 1,
            "completed_at_utc": utc_now(),
            "consumption_marker": artifact(marker),
            "evaluation_forward_count": 1,
            "head_count_retained": 1,
            "layer_count_retained": 1,
            "record_count_retained": 1,
            "token_count_retained": 1,
        },
        "coordinate": {
            "dataset": "c4_en_512",
            "dataset_record_index": 64,
            "head": HEAD,
            "layer": 0,
            "token": TOKEN,
            "token_id": live_binding["evaluation_witness"]["token_id"],
        },
        "input_binding": live_binding,
        "model": {
            "repository": model_spec["repository"],
            "revision": model_spec["revision"],
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "dtype": "bfloat16_source_with_fixed_point_layer0_capture",
        },
        "kv_write": {
            "head_order": ["k_head0", "k_head1", "v_head0", "v_head1"],
            "joint_metadata_encoding": "four little-endian normalized Scale32 u32 records",
            "joint_metadata_records_u32": records,
            "k_head1_saturated_elements": int(k_saturations),
            "required_shell_payload_bytes": {"k": 128, "v": 128, "metadata": 16},
        },
        "tensor_manifest": tensors,
        "source_bindings": {
            "capture_tool": artifact(source_archive),
            "fixed_model": artifact(ROOT / "tools/ace2_full_model_fixed_point.py"),
            "head0_sha256s": artifact(HEAD0_SUMS),
            "maintained_shell": artifact(ROOT / "rtl/ace2_shell.sv"),
            "maintained_shell_testbench": artifact(ROOT / "verification/tb/ace2_shell_tb.sv"),
            "rtl_tree": base.rtl_tree(),
        },
        "runtime": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": versions,
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "scope_exclusions": [
            "head0_recapture",
            "other_records",
            "other_tokens",
            "other_heads",
            "other_layers",
            "quality_baseline_candidate_benchmark_scale32",
            "synthesis_opensta_ppa",
            "rtl_mutation",
            "frontier_authority_seal_mutation",
        ],
    }
    write_json(CAPTURE / "witness.json", witness)
    write_json(
        CAPTURE / "capture_contract.json",
        {
            "schema_version": 1,
            "status": "CAPTURE_COMPLETE_MERGE_REPLAY_PENDING",
            "completed_at_utc": utc_now(),
            "coordinate": witness["coordinate"],
            "evaluation_forward_count": 1,
            "source": artifact(source_archive),
            "tensor_count": len(tensors),
            "witness": artifact(CAPTURE / "witness.json"),
        },
    )
    (CAPTURE / "capture.log").write_text(
        "ACE2_KV_HEAD1_REFERENCE_CAPTURE_COMPLETE\n"
        "coordinate=c4_en_512:record64:layer0:token1:head1\n"
        "evaluation_forward_count=1\n"
        "retained_head_count=1\n"
        "tensor_count=3\n"
        "head0_recaptured=false\n"
        "rtl_mutated=false\n",
        encoding="utf-8",
    )
    print(
        "ACE2_KV_HEAD1_REFERENCE_CAPTURE_COMPLETE "
        f"tensors=3 evaluation_forwards=1 witness_sha256={sha256_file(CAPTURE / 'witness.json')}"
    )


def merge_packages() -> None:
    require(not MERGE.exists(), "merge output already exists")
    h0 = head0_witness()
    validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE)
    h1 = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    MERGE.mkdir(parents=True)
    logical = {
        "k_head0": h0["tensor_manifest"]["kv_write_k_head0_token1_s8"],
        "k_head1": h1["tensor_manifest"]["kv_write_k_head1_token1_s8"],
        "v_head0": h0["tensor_manifest"]["kv_write_v_head0_token1_s8"],
        "v_head1": h1["tensor_manifest"]["kv_write_v_head1_token1_s8"],
        "joint_metadata": h1["tensor_manifest"]["kv_write_joint_two_head_scale_metadata_u8"],
    }
    write_json(
        MERGE / "merge_manifest.json",
        {
            "schema_version": 1,
            "mission_id": "rtl-kv-write-two-head-reference-batch-replay-v1",
            "classification": "derived_hash_bound_logical_merge_no_source_package_modification",
            "coordinate": {"dataset": "c4_en_512", "dataset_record_index": 64, "layer": 0, "token": 1},
            "source_packages": {
                "immutable_head0": artifact(HEAD0_SUMS),
                "minimal_head1": artifact(CAPTURE / "SHA256SUMS"),
            },
            "logical_kv_write_interface": logical,
            "counts": {"k_bytes": 128, "v_bytes": 128, "metadata_bytes": 16, "kv_heads": 2},
            "source_packages_modified": False,
        },
    )
    write_sha256s(MERGE)


def tensor_raw(record: dict[str, Any]) -> bytes:
    path = ROOT / record["path"]
    raw = path.read_bytes()
    require(len(raw) == record["bytes"], f"tensor byte count changed: {path}")
    require(hashlib.sha256(raw).hexdigest() == record["sha256"], f"tensor hash changed: {path}")
    return raw


def render_kv_include(k_raw: bytes, v_raw: bytes, metadata: bytes) -> str:
    require(len(k_raw) == 128 and len(v_raw) == 128 and len(metadata) == 16, "KV include shape changed")
    streams = [k_raw, v_raw, metadata]
    lines = [
        "function automatic [127:0] c4_kv_word;",
        "    input integer stream_id;",
        "    input integer beat;",
        "    begin",
        "        c4_kv_word = 128'd0;",
        "        case (stream_id)",
    ]
    for stream_id, raw in enumerate(streams):
        lines.append(f"            {stream_id}: begin")
        lines.append("                case (beat)")
        for beat in range(len(raw) // 16):
            value = int.from_bytes(raw[beat * 16 : (beat + 1) * 16], "little")
            lines.append(f"                    {beat}: c4_kv_word = 128'h{value:032x};")
        lines.extend(["                    default: c4_kv_word = 128'd0;", "                endcase", "            end"])
    lines.extend(["            default: c4_kv_word = 128'd0;", "        endcase", "    end", "endfunction", ""])
    return "\n".join(lines)


def execution_testbench() -> str:
    text = (ROOT / "verification/tb/ace2_shell_tb.sv").read_text(encoding="utf-8")
    function_pattern = re.compile(r"    function \[127:0\] make_kv_word;.*?    endfunction", re.DOTALL)
    replacement = (
        '    `include "c4_kv_write_witness.svh"\n\n'
        "    function [127:0] make_kv_word;\n"
        "        input integer selected_case;\n"
        "        input integer stream_id;\n"
        "        input integer beat;\n"
        "        begin\n"
        "            make_kv_word = c4_kv_word(stream_id, beat);\n"
        "        end\n"
        "    endfunction"
    )
    text, count = function_pattern.subn(replacement, text, count=1)
    require(count == 1, "maintained KV data function changed")
    text = text.replace("    integer rope_only_mode;", "    integer rope_only_mode;\n    integer kv_write_only_mode;", 1)
    text = text.replace(
        '        rope_only_mode = $test$plusargs("ROPE_ONLY");',
        '        rope_only_mode = $test$plusargs("ROPE_ONLY");\n'
        '        kv_write_only_mode = $test$plusargs("KV_WRITE_ONLY");',
        1,
    )
    branch = (
        "        if (kv_write_only_mode) begin\n"
        "            send_kv_write_cmd(0, 0, 16'd1, 16'h3b00);\n"
        "            wait_kv_write_done_and_compare(0, 0, 16'h3b00);\n"
        "            if (failures != 0) begin\n"
        "                $display(\"ACE2_C4_KV_WRITE_REPLAY_FAIL failures=%0d\", failures);\n"
        "                $fatal(1, \"ACE2_C4_KV_WRITE_REPLAY_FAIL\");\n"
        "            end\n"
        "            $display(\"ACE2_C4_KV_WRITE_REPLAY_PASS writes=%0d cycles=%0d\", observed_count, last_command_cycles);\n"
        "            $finish;\n"
        "        end\n\n"
    )
    needle = "        if (smoke_mode) begin"
    require(needle in text, "maintained shell focused-mode insertion point changed")
    text = text.replace(needle, branch + needle, 1)
    require("kv_write_only_mode" in text, "KV_WRITE focused mode was not inserted")
    return text


def run_replay() -> None:
    if REPLAY.exists():
        require(not (REPLAY / "SHA256SUMS").exists(), "sealed replay output already exists")
        require(not (REPLAY / "ordered_prefix_report.json").exists(), "completed replay report already exists")
        retained = [
            path
            for path in (REPLAY / "iverilog.log", REPLAY / "replay.log", REPLAY / "replay.stdout", REPLAY / "replay.stderr", REPLAY / "replay_tb_source_at_execution.sv", REPLAY / "generated")
            if path.exists()
        ]
        if retained:
            attempts = REPLAY / "failed_attempts"
            attempts.mkdir(exist_ok=True)
            attempt_index = 1
            while (attempts / f"attempt-{attempt_index:04d}").exists():
                attempt_index += 1
            attempt = attempts / f"attempt-{attempt_index:04d}"
            attempt.mkdir()
            for path in retained:
                shutil.move(str(path), str(attempt / path.name))
    else:
        REPLAY.mkdir(parents=True)
    merge = json.loads((MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    logical = merge["logical_kv_write_interface"]
    k_raw = tensor_raw(logical["k_head0"]) + tensor_raw(logical["k_head1"])
    v_raw = tensor_raw(logical["v_head0"]) + tensor_raw(logical["v_head1"])
    metadata = tensor_raw(logical["joint_metadata"])
    generated = REPLAY / "generated"
    generated.mkdir()
    (generated / "c4_kv_write_witness.svh").write_text(render_kv_include(k_raw, v_raw, metadata), encoding="utf-8")
    shutil.copy2(SELF, REPLAY / "replay_source_at_execution.py")
    tb = REPLAY / "replay_tb_source_at_execution.sv"
    tb.write_text(execution_testbench(), encoding="utf-8")

    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    image = ROOT / "build/rtl-kv-write-two-head-reference-batch-replay-v1/kv_write.vvp"
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
    require(compiled.returncode == 0, "KV_WRITE replay compilation failed")
    replay_command = ["vvp", str(image), "+KV_WRITE_ONLY"]
    replayed = subprocess.run(replay_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (REPLAY / "replay.stdout").write_text(replayed.stdout, encoding="utf-8")
    (REPLAY / "replay.stderr").write_text(replayed.stderr, encoding="utf-8")
    (REPLAY / "replay.log").write_text(
        "command=" + " ".join(replay_command) + "\n" + replayed.stdout + replayed.stderr,
        encoding="utf-8",
    )
    require(replayed.returncode == 0, "KV_WRITE replay failed")
    match = re.search(r"ACE2_C4_KV_WRITE_REPLAY_PASS writes=(\d+) cycles=(\d+)", replayed.stdout)
    require(match is not None, "KV_WRITE replay pass marker missing")
    require(int(match.group(1)) == 17, "KV_WRITE write count changed")
    require("MISMATCH" not in replayed.stdout and "_FAIL" not in replayed.stdout, "KV_WRITE replay mismatch reported")

    boundary = {
        "schema_version": 1,
        "mission_id": "rtl-kv-write-two-head-reference-batch-replay-v1",
        "operator": "layer_0.attention_score",
        "classification": "UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_RETAINED_COORDINATE_INCOMPLETE",
        "available_inputs": {
            "k_cache_head0_token1_s8": logical["k_head0"],
            "k_cache_head1_token1_s8": logical["k_head1"],
            "attention_score_multiplier_s32": head0_witness()["operator_metadata"]["attention_score_multiplier_s32"],
            "attention_score_right_shift_u6": head0_witness()["operator_metadata"]["attention_score_right_shift_u6"],
        },
        "missing_fields": [
            "layer_0.attention_score.input.query_head0_token1_s8[64]",
            "layer_0.attention_score.input.k_cache_head0_token0_s8[64]",
        ],
        "scope_reason": "The authorized merge contains only the immutable head-0 package and minimal head-1 package; neither retains token-0 K-cache or the RoPE-Q payload.",
        "simulation_executed": False,
        "later_operators_attempted": False,
    }
    write_json(REPLAY / "boundary_probe.json", boundary)
    report = {
        "schema_version": 1,
        "mission_id": "rtl-kv-write-two-head-reference-batch-replay-v1",
        "producer_role": "engineer",
        "review_status": "PENDING_FRESH_REVIEWER",
        "coordinate": {"dataset": "c4_en_512", "dataset_record_index": 64, "layer": 0, "token": 1},
        "capture": {
            "head0_reused_not_recaptured": True,
            "head0_tensor_count": 30,
            "head1_evaluation_forward_count": 1,
            "head1_tensor_count": 3,
            "consumption_marker": artifact(CAPTURE / "capture_execution_consumed.json"),
            "merge_manifest": artifact(MERGE / "merge_manifest.json"),
        },
        "frozen_order": ORDER,
        "rtl_tree": base.rtl_tree(),
        "prior_maximal_prefix": ORDER[:6],
        "attempted_operators": [
            {
                "operator": "layer_0.kv_write",
                "status": "PASS_EXACT",
                "input_hashes": {
                    "k_head0": logical["k_head0"]["sha256"],
                    "k_head1": logical["k_head1"]["sha256"],
                    "v_head0": logical["v_head0"]["sha256"],
                    "v_head1": logical["v_head1"]["sha256"],
                    "joint_metadata": logical["joint_metadata"]["sha256"],
                },
                "output_hashes": {
                    "k_cache_payload": hashlib.sha256(k_raw).hexdigest(),
                    "v_cache_payload": hashlib.sha256(v_raw).hexdigest(),
                    "metadata_payload": hashlib.sha256(metadata).hexdigest(),
                },
                "rtl_source_hashes": {
                    "shell": artifact(ROOT / "rtl/ace2_shell.sv"),
                    "package": artifact(ROOT / "rtl/ace2_pkg.sv"),
                    "maintained_harness": artifact(ROOT / "verification/tb/ace2_shell_tb.sv"),
                    "execution_copy": artifact(tb),
                    "replay_tool": artifact(REPLAY / "replay_source_at_execution.py"),
                },
                "counts": {
                    "k_bytes": 128,
                    "v_bytes": 128,
                    "metadata_bytes": 16,
                    "writes": 17,
                    "compared_beats": 17,
                    "mismatches": 0,
                },
                "saturation": {"result": "NOT_APPLICABLE_COPY_OPERATOR"},
                "cycle_handshake_latency": {
                    "simulation_executed": True,
                    "total_command_cycles": int(match.group(2)),
                    "handshake_result": "PASS_NO_TIMEOUT_OR_DROPPED_BEAT",
                    "latency_result": "PASS_BOUNDED_BY_MAINTAINED_SHELL_WATCHDOG",
                },
                "raw_log": artifact(REPLAY / "replay.log"),
            },
            {
                "operator": "layer_0.attention_score",
                "status": boundary["classification"],
                "counts": {"missing_input_count": len(boundary["missing_fields"]), "mismatch_count": None},
                "cycle_handshake_latency": {
                    "simulation_executed": False,
                    "handshake_result": "NOT_EVALUATED_INTERFACE_UNAVAILABLE",
                    "latency_result": "NOT_EVALUATED_INTERFACE_UNAVAILABLE",
                },
                "boundary_probe": artifact(REPLAY / "boundary_probe.json"),
            },
        ],
        "newly_passing_operators": ["layer_0.kv_write"],
        "maximal_passing_prefix": ORDER[:7],
        "first_failing_or_unsupported_operator": {
            "operator": "layer_0.attention_score",
            "classification": boundary["classification"],
            "boundary_probe": artifact(REPLAY / "boundary_probe.json"),
        },
        "batch_result": "STOPPED_AT_FIRST_PRECISELY_UNAVAILABLE_EXECUTABLE_INTERFACE",
        "later_operators_attempted": False,
        "existing_current_tree_ppa": {"reused_not_rerun": True, "sha256_manifest": artifact(PPA_SUMS)},
        "forbidden_flows_run": [],
        "state_mutations": {
            "rtl": False,
            "frontier_ledger_manifest_traceability_pipeline_seals": False,
            "synthesis_opensta_ppa": False,
            "baseline_candidate_benchmark_scale32": False,
        },
        "review_requirement": "Fresh Reviewer must independently accept or reject this maximal-prefix result.",
    }
    write_json(REPLAY / "ordered_prefix_report.json", report)
    (REPLAY / "batch_replay.log").write_text(
        "ACE2_KV_WRITE_TWO_HEAD_ORDERED_BATCH_REPLAY\n"
        f"rtl_tree_sha256={EXPECTED_RTL_TREE}\n"
        "coordinate=c4_en_512:record64:layer0:token1:heads0-1\n"
        "layer_0.kv_write=PASS_EXACT writes=17 mismatches=0\n"
        "layer_0.attention_score=UNAVAILABLE_EXACT_EXECUTABLE_INTERFACE_RETAINED_COORDINATE_INCOMPLETE\n"
        "maximal_passing_operator=layer_0.kv_write\n"
        "later_operators_attempted=false\n"
        "synthesis_opensta_ppa_executed=false\n",
        encoding="utf-8",
    )
    write_sha256s(REPLAY)
    print(
        "ACE2_KV_WRITE_TWO_HEAD_BATCH_REPLAY_BOUNDARY "
        "maximal_prefix=layer_0.kv_write first_boundary=layer_0.attention_score mismatches=0"
    )


def launch_capture_and_replay() -> None:
    preflight = validate_preflight()
    require(not (CAPTURE / "capture_execution_consumed.json").exists(), "capture already consumed")
    require(not (CAPTURE / "witness.json").exists(), "capture witness already exists")
    require(not MERGE.exists(), "merge output already exists")
    require(not REPLAY.exists(), "replay output already exists")
    source_archive = CAPTURE / "capture_source_at_execution.py"
    require(not source_archive.exists(), "capture source archive already exists")
    shutil.copy2(SELF, source_archive)
    require(sha256_file(source_archive) == preflight["source"]["sha256"], "source changed after preflight")
    stdout_path = CAPTURE / "capture.stdout"
    stderr_path = CAPTURE / "capture.stderr"
    marker = CAPTURE / "capture_execution_consumed.json"
    with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open("x", encoding="utf-8") as stderr_handle:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "mission_id": "rtl-kv-write-two-head-reference-batch-replay-v1",
                    "consumed_at_utc": utc_now(),
                    "command": [sys.executable, relative(SELF), "_capture_worker"],
                    "coordinate": {"dataset": "c4_en_512", "dataset_record_index": 64, "head": 1, "layer": 0, "token": 1},
                    "forward_capture_limit": 1,
                    "rerun_permitted": False,
                    "raw_stdout": relative(stdout_path),
                    "raw_stderr": relative(stderr_path),
                    "source_sha256": preflight["source"]["sha256"],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        worker = subprocess.run(
            [sys.executable, str(SELF), "_capture_worker"],
            cwd=ROOT,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    write_json(
        CAPTURE / "capture_worker_status.json",
        {"schema_version": 1, "completed_at_utc": utc_now(), "returncode": worker.returncode},
    )
    require(worker.returncode == 0, "single capture worker failed; authorization remains consumed")
    write_sha256s(CAPTURE)
    merge_packages()
    run_replay()


def run_check() -> None:
    validate_preflight()
    head0_witness()
    validate_manifest(CAPTURE / "SHA256SUMS", CAPTURE)
    validate_manifest(MERGE / "SHA256SUMS", MERGE)
    validate_manifest(REPLAY / "SHA256SUMS", REPLAY)
    marker = json.loads((CAPTURE / "capture_execution_consumed.json").read_text(encoding="utf-8"))
    witness = json.loads((CAPTURE / "witness.json").read_text(encoding="utf-8"))
    merge = json.loads((MERGE / "merge_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((REPLAY / "ordered_prefix_report.json").read_text(encoding="utf-8"))
    require(marker["rerun_permitted"] is False, "consumption marker permits rerun")
    require(witness["capture"]["evaluation_forward_count"] == 1, "head-1 capture was not exactly once")
    require(witness["coordinate"]["head"] == 1, "head-1 coordinate changed")
    require(len(witness["tensor_manifest"]) == 3, "head-1 tensor count changed")
    require(merge["source_packages_modified"] is False, "merge claims source-package modification")
    require(report["newly_passing_operators"] == ["layer_0.kv_write"], "newly passing operator changed")
    require(report["maximal_passing_prefix"] == ORDER[:7], "maximal prefix changed")
    require(report["first_failing_or_unsupported_operator"]["operator"] == "layer_0.attention_score", "first boundary changed")
    require(report["attempted_operators"][0]["counts"]["mismatches"] == 0, "KV_WRITE mismatches appeared")
    require(base.rtl_tree()["sha256"] == EXPECTED_RTL_TREE, "RTL tree changed after replay")
    require(sha256_file(HEAD0_SUMS) == EXPECTED_HEAD0_SUMS, "head-0 source package changed")
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SUMS, "canonical PPA aggregate changed")
    print(
        "ACE2_KV_WRITE_TWO_HEAD_CAPTURE_REPLAY_CHECK_PASS "
        f"capture_sha256={sha256_file(CAPTURE / 'witness.json')} "
        f"merge_sha256={sha256_file(MERGE / 'merge_manifest.json')} "
        f"report_sha256={sha256_file(REPLAY / 'ordered_prefix_report.json')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "capture-and-replay", "check", "replay-only", "_capture_worker"))
    args = parser.parse_args()
    if args.mode == "preflight":
        run_preflight()
    elif args.mode == "capture-and-replay":
        launch_capture_and_replay()
    elif args.mode == "replay-only":
        require((CAPTURE / "SHA256SUMS").is_file(), "completed capture package missing")
        if not MERGE.exists():
            merge_packages()
        run_replay()
    elif args.mode == "check":
        run_check()
    else:
        execute_capture_worker()


if __name__ == "__main__":
    main()
