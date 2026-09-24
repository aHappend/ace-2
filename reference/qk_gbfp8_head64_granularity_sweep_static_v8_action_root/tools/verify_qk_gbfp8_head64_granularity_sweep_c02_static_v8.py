#!/usr/bin/env python3
"""Focused decisive check for the V8 dual-c02-reader increment."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import platform
import stat
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


sys.dont_write_bytecode = True
ACTION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ACTION_ROOT.parents[1]
PARSER_PATH = ACTION_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
MANIFEST_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_INCREMENT.json"
V7_ROOT = REPOSITORY_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v7_action_root"
V6_ROOT = REPOSITORY_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root"
V6_HANDOFF = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/c9e0569b083e")
SEALED_TENSOR = V6_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"

V7_IDENTITIES = {
    "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_MANIFEST.json": "e06e83f9a939d005d86bd1d428fa1ccfa3c41624e561a622e0a788d8aa682a7a",
    "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_RUNTIME_PACKAGE.json": "9b908efdcd78d646c7c69dbf364f02f050cc507492666cfe38892ccd4380498d",
    "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_PACKAGE.json": "424c16dd9a352158c13518b077224eca7b05ec87509e0e1c7afe6e8088cdd102",
    "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V7_RESULT_SCHEMA.json": "6be8515a19952f85776af525b234dd5db8bb77df86cd205becc101888670f3d7",
    "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v7.py": "565dd9727458199157f8eba0edb3aa39a13d78bfc076f259433022efc49d9474",
    "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v7.py": "13eb6348aeb180722c1addee3220454119fbb4a428c36a63d5f8416759feac1e",
    "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v7_schema_compatibility.py": "a8c45cce3cdf740ee1e3a12ef226a4bacda394835107b45160552ff5a37b2237",
}
V6_IDENTITIES = {
    V6_HANDOFF / "round-0001.json": "0d20e696e77e47e425a761d726fa10ab57c09733d7d3af066ea6f874b7debd49",
    V6_HANDOFF / "v6ns-r2-aftermath-recompute.json": "7f17929b5b087764e3d321b8354e750ac42542aad772fe18cb4f4942e01b53fb",
    V6_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v6/base/result.json": "74b5d8fded1e344848023144336d0f8de8e0409c6f589fb8c9a7b642fade695c",
    V6_ROOT / "build/qk-gbfp8-g16-head64-successor-diagnostic-v6-authority/base/first-terminal.json": "83a875fab228f552e6b26cfeacf021296ad65b122981873570d65ea9c4f367e6",
}
LIVE_V8_PATHS = (
    REPOSITORY_ROOT / "build/qk-gbfp8-head64-granularity-sweep-v8/base/result.json",
    REPOSITORY_ROOT / "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/authority.json",
    REPOSITORY_ROOT / "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/credential.json",
    REPOSITORY_ROOT / "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/authority-ledger.json",
    REPOSITORY_ROOT / "build/qk-gbfp8-head64-granularity-sweep-v8-authority/base/first-terminal.json",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    require(os.path.abspath(path) != os.path.abspath(SEALED_TENSOR), "sealed tensor hash prohibited")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(VerificationError(f"nonfinite JSON: {token}")),
    )
    require(type(value) is dict, "manifest object")
    return value


def load_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("qk_gbfp8_v8_c02", PARSER_PATH)
    require(specification is not None and specification.loader is not None, "parser module spec")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def oracle_digest(dtype: str, shape: tuple[int, ...], payload: bytes) -> str:
    framed = dtype.encode("ascii") + b"\0" + struct.pack(">I", len(shape))
    framed += b"".join(struct.pack(">Q", dimension) for dimension in shape)
    return hashlib.sha256(framed + payload).hexdigest()


def binding(dtype: str, shape: tuple[int, ...], payload: bytes) -> dict[str, Any]:
    return {"dtype": dtype, "sha256": oracle_digest(dtype, shape, payload), "shape": list(shape)}


def raw_bundle(records: list[tuple[bytes, bytes, tuple[int, ...], bytes, int | None]], count: int | None = None) -> bytes:
    output = bytearray(b"ACE2-C02-TENSORS-V1\n")
    output.extend(struct.pack(">I", len(records) if count is None else count))
    for name, dtype, shape, payload, declared_size in records:
        output.extend(struct.pack(">H", len(name)))
        output.extend(name)
        output.extend(struct.pack("B", len(dtype)))
        output.extend(dtype)
        output.extend(struct.pack("B", len(shape)))
        for dimension in shape:
            output.extend(struct.pack(">Q", dimension))
        output.extend(struct.pack(">Q", len(payload) if declared_size is None else declared_size))
        output.extend(payload)
    return bytes(output)


def expect_both_reject(module: ModuleType, data: bytes, bindings: dict[str, dict[str, Any]], label: str) -> None:
    for parser_name in ("parse_c02_producer_bytes", "parse_c02_accepted_reader"):
        parser = getattr(module, parser_name)
        try:
            parser(data, bindings)
        except module.C02Error:
            continue
        except Exception as error:
            raise VerificationError(f"{label}: {parser_name} leaked {type(error).__name__}") from error
        raise VerificationError(f"{label}: {parser_name} accepted mutation")


def verify_parser_independence(module: ModuleType, bundle: bytes, bindings: dict[str, dict[str, Any]]) -> None:
    source = PARSER_PATH.read_text("utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    producer = functions["parse_c02_producer_bytes"]
    accepted = functions["parse_c02_accepted_reader"]
    producer_names = {node.id for node in ast.walk(producer) if isinstance(node, ast.Name)}
    accepted_names = {node.id for node in ast.walk(accepted) if isinstance(node, ast.Name)}
    require("parse_c02_accepted_reader" not in producer_names, "producer delegates to accepted reader")
    require("parse_c02_producer_bytes" not in accepted_names, "accepted reader delegates to producer")
    require(not any(name.startswith("_accepted") for name in producer_names), "producer uses accepted-reader helper")
    require(not any(name.startswith("_producer") for name in accepted_names), "accepted reader uses producer helper")

    original_producer = module.parse_c02_producer_bytes
    original_accepted = module.parse_c02_accepted_reader

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise VerificationError("cross-parser delegation")

    module.parse_c02_producer_bytes = forbidden
    try:
        require(original_accepted(bundle, bindings), "accepted-reader independence runtime")
    finally:
        module.parse_c02_producer_bytes = original_producer
    module.parse_c02_accepted_reader = forbidden
    try:
        require(original_producer(bundle, bindings), "producer-parser independence runtime")
    finally:
        module.parse_c02_accepted_reader = original_accepted


def verify_manifest(manifest: dict[str, Any]) -> None:
    require(
        set(manifest)
        == {
            "artifact_bindings",
            "artifact_kind",
            "claim_boundary",
            "frozen_inputs",
            "grammar_contract",
            "increment_scope",
            "mission_id",
            "root_id",
            "schema_version",
            "toolchain",
        },
        "manifest exact keys",
    )
    require(
        manifest["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_static_v8_c02_component",
        "manifest artifact kind",
    )
    require(manifest["mission_id"] == "f0fa5c269681" and manifest["schema_version"] == 1, "manifest identity")
    require(manifest["root_id"] == ACTION_ROOT.name, "manifest root identity")
    require(manifest["increment_scope"] == "SORTED_DUAL_C02_READERS_COMPONENT_OF_FULL_BASE", "manifest bounded scope")
    require(
        manifest["claim_boundary"]
        == {
            "authority_materialized": False,
            "checkpoint_176_activity": False,
            "controller_or_evaluator_process_executed": False,
            "credential_materialized": False,
            "fresh_l2_accepted": False,
            "hardware_or_rtl_activity": False,
            "ledger_result_or_terminal_materialized": False,
            "sealed_tensor_opened_read_or_hashed": False,
            "static_acceptance_grants_execution_authority": False,
            "tensor_projection_performed": False,
        },
        "manifest claim boundary",
    )
    require(
        manifest["artifact_bindings"]
        == [
            {"path": "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py", "sha256": sha256_file(PARSER_PATH)},
            {"path": "tools/verify_qk_gbfp8_head64_granularity_sweep_c02_static_v8.py", "sha256": sha256_file(Path(__file__))},
        ],
        "manifest artifact bindings",
    )
    frozen = manifest["frozen_inputs"]
    require(frozen["v7_files"] == V7_IDENTITIES, "manifest V7 identities")
    require(frozen["v6_files"] == {path.name: digest for path, digest in V6_IDENTITIES.items()}, "manifest V6 identities")
    require(frozen["sealed_tensor_lstat"] == {"byte_count": 1305797, "file_type": "regular", "mode_octal": "0400"}, "manifest sealed lstat")
    require(
        manifest["grammar_contract"]
        == {
            "accepted_reader_implementation": "BytesIO plus Struct with independent binding, digest, geometry, nonfinite, and EOF checks",
            "digest": "sha256(dtype_ascii || NUL || rank_u32_be || dimensions_u64_be || payload)",
            "full_record_bf16_nonfinite_rejection_before_return": True,
            "magic": "ACE2-C02-TENSORS-V1\n",
            "payload_boundary": "declared bytes exact; BF16 bytes equal 2*product(shape)",
            "producer_parser_implementation": "cursor plus int.from_bytes with independent binding, digest, geometry, nonfinite, and EOF checks",
            "record": "name_u16_be+utf8, dtype_u8+ascii, rank_u8, dimensions_u64_be, payload_size_u64_be, payload",
            "record_name_order": "strictly increasing Unicode tensor name order matching producer sorted(tensors)",
            "strict_eof": True,
        },
        "manifest grammar contract",
    )
    toolchain = manifest["toolchain"]
    require(toolchain["interpreter_path"] == "/home/argustest/miniconda3/bin/python3.13", "interpreter path binding")
    require(os.path.abspath(sys.executable) == toolchain["interpreter_path"], "running interpreter path")
    require(platform.python_version() == toolchain["interpreter_version"] == "3.13.5", "interpreter version binding")
    require(sha256_file(Path(sys.executable)) == toolchain["interpreter_sha256"], "interpreter checksum binding")
    require(toolchain["verification_mode"] == "synthetic_bytes_static_only", "verification mode")


def main() -> int:
    sealed_open_events: list[tuple[Any, ...]] = []

    def audit_hook(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if isinstance(candidate, (str, bytes, os.PathLike)) and os.path.abspath(os.fsdecode(candidate)) == os.path.abspath(SEALED_TENSOR):
            sealed_open_events.append(arguments)
            raise VerificationError("sealed tensor open/read/hash prohibited")

    sys.addaudithook(audit_hook)
    manifest = load_json(MANIFEST_PATH)
    module = load_module()
    verify_manifest(manifest)

    for relative, expected in V7_IDENTITIES.items():
        path = V7_ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"frozen V7 file type: {relative}")
        require(sha256_file(path) == expected, f"frozen V7 identity: {relative}")
    for path, expected in V6_IDENTITIES.items():
        require(path.is_file() and not path.is_symlink(), f"frozen V6 file type: {path.name}")
        require(sha256_file(path) == expected, f"frozen V6 identity: {path.name}")
    tensor_stat = os.lstat(SEALED_TENSOR)
    require(stat.S_ISREG(tensor_stat.st_mode), "sealed tensor regular file")
    require(tensor_stat.st_size == 1305797 and stat.S_IMODE(tensor_stat.st_mode) == 0o400, "sealed tensor lstat identity")
    require(not sealed_open_events, "sealed tensor opened")
    require(all(not os.path.lexists(path) for path in LIVE_V8_PATHS), "live V8 artifact exists")

    words = (0x3F80, 0x3F00, 0xBF00, 0x0000, *([0x3C00] * 60))
    payload = struct.pack("<64H", *words)
    records = [
        {"dtype": "torch.bfloat16", "name": "query", "payload": payload, "shape": (1, 1, 64)},
        {"dtype": "torch.bfloat16", "name": "key", "payload": payload, "shape": (1, 1, 64)},
    ]
    bindings = {record["name"]: binding(record["dtype"], record["shape"], record["payload"]) for record in records}
    bundle = module.produce_c02_bundle(records)
    require(bundle == module.produce_c02_bundle(list(reversed(records))), "fixture producer canonical order")
    producer_records = module.parse_c02_producer_bytes(bundle, bindings)
    accepted_records = module.parse_c02_accepted_reader(bundle, bindings)
    require(producer_records == accepted_records, "cross-parser canonical agreement")
    require(tuple(producer_records) == ("key", "query"), "record order")
    require(producer_records["query"] == {"dtype": "torch.bfloat16", "payload": payload, "sha256": bindings["query"]["sha256"], "shape": (1, 1, 64)}, "canonical parsed record")
    verify_parser_independence(module, bundle, bindings)

    mutations: list[tuple[str, bytes, dict[str, dict[str, Any]]]] = []
    mutations.append(("magic", bytes((bundle[0] ^ 1,)) + bundle[1:], bindings))
    mutations.append(("truncated_eof", bundle[:-1], bindings))
    mutations.append(("trailing_eof", bundle + b"\0", bindings))
    mutations.append(("zero_count", module.C02_MAGIC + b"\0\0\0\0", {}))
    unsorted_bytes = raw_bundle([
        (b"z", b"torch.bfloat16", (1, 1, 64), payload, None),
        (b"a", b"torch.bfloat16", (1, 1, 64), payload, None),
    ])
    unsorted_bindings = {
        "a": binding("torch.bfloat16", (1, 1, 64), payload),
        "z": binding("torch.bfloat16", (1, 1, 64), payload),
    }
    mutations.append(("record_name_order", unsorted_bytes, unsorted_bindings))
    duplicate_payload = payload
    duplicate_bytes = raw_bundle([
        (b"same", b"torch.bfloat16", (1, 1, 64), duplicate_payload, None),
        (b"same", b"torch.bfloat16", (1, 1, 64), duplicate_payload, None),
    ])
    mutations.append(("duplicate_name", duplicate_bytes, {"same": binding("torch.bfloat16", (1, 1, 64), duplicate_payload)}))
    invalid_name = raw_bundle([(b"\xff", b"torch.bfloat16", (1, 1, 64), payload, None)])
    mutations.append(("name_utf8", invalid_name, {}))
    invalid_dtype = raw_bundle([(b"query", b"\xff", (1, 1, 64), payload, None)])
    mutations.append(("dtype_ascii", invalid_dtype, {}))
    rank_zero = raw_bundle([(b"query", b"torch.bfloat16", (), b"", None)])
    mutations.append(("rank_zero", rank_zero, {}))
    dimension_zero = raw_bundle([(b"query", b"torch.bfloat16", (1, 0, 64), payload, None)])
    mutations.append(("dimension_zero", dimension_zero, {}))
    declared_too_large = raw_bundle([(b"query", b"torch.bfloat16", (1, 1, 64), b"", (1 << 31) + 1)])
    mutations.append(("declared_payload_limit", declared_too_large, {}))
    declared_overrun = raw_bundle([(b"query", b"torch.bfloat16", (1, 1, 64), payload, len(payload) + 1)])
    mutations.append(("declared_payload_overrun", declared_overrun, {"query": bindings["query"]}))
    short_payload = payload[:-2]
    short_bundle = module.produce_c02_bundle([{"dtype": "torch.bfloat16", "name": "query", "payload": short_payload, "shape": (1, 1, 64)}])
    mutations.append(("bf16_payload_boundary", short_bundle, {"query": binding("torch.bfloat16", (1, 1, 64), short_payload)}))
    nonfinite_payload = payload[:-2] + struct.pack("<H", 0x7F80)
    nonfinite_bundle = module.produce_c02_bundle([{"dtype": "torch.bfloat16", "name": "query", "payload": nonfinite_payload, "shape": (1, 1, 64)}])
    mutations.append(("full_record_nonfinite", nonfinite_bundle, {"query": binding("torch.bfloat16", (1, 1, 64), nonfinite_payload)}))
    bad_digest = {name: dict(value) for name, value in bindings.items()}
    bad_digest["query"]["sha256"] = "0" * 64
    mutations.append(("binding_digest", bundle, bad_digest))
    payload_mutation = bundle[:-1] + bytes((bundle[-1] ^ 1,))
    mutations.append(("payload_digest", payload_mutation, bindings))
    missing_binding = {"query": bindings["query"]}
    mutations.append(("missing_binding", bundle, missing_binding))
    extra_binding = dict(bindings)
    extra_binding["extra"] = binding("torch.bfloat16", (1, 1, 64), payload)
    mutations.append(("extra_binding", bundle, extra_binding))

    for label, malformed, malformed_bindings in mutations:
        expect_both_reject(module, malformed, malformed_bindings, label)

    require(not sealed_open_events, "sealed tensor opened during parser checks")
    print(
        "PASS_V8_C02_STATIC_INCREMENT "
        f"grammar_and_guard_mutations={len(mutations) * 2} "
        "cross_parsers=2 sealed_tensor_access=lstat_only "
        f"frozen_v7_files={len(V7_IDENTITIES)} frozen_v6_files={len(V6_IDENTITIES)} live_v8_artifacts=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
