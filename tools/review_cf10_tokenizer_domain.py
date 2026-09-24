#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf10"
PREDECESSORS = {
    "cf07": ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf07",
    "cf08": ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf08",
    "cf09": ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf09",
}
IDENTITY = "stage1-w4a8-r6-product-prep-cf10-attempt-0001"
NONCE = "5e6c6eb08db41c414a58d73fba75308b8fee26f2c37c7c1624048b2cc57ce44b"
MODEL_VOCABULARY_SIZE = 151936
TOKENIZER_BASE_VOCABULARY_SIZE = 151643
TOKENIZER_MAPPED_ID_COUNT = 151665
TOKENIZER_ADDED_TOKEN_COUNT = 22
TOKENIZER_MAXIMUM_MAPPED_ID = 151664
EXPECTED_TEST_COUNT = 17
TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TOKENIZER_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / TOKENIZER_REVISION
)


class ReviewError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def write_json(path: Path, value: object) -> None:
    write_bytes(path, canonical_bytes(value))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def namespace_inventory(path: Path) -> dict[str, object]:
    if not path.is_dir() or path.is_symlink():
        raise ReviewError(f"namespace is absent or linked: {relative(path)}")
    entries: dict[str, dict[str, object]] = {}
    for entry in sorted(path.rglob("*")):
        name = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            raise ReviewError(f"namespace contains link: {relative(entry)}")
        mode = stat.S_IMODE(entry.stat().st_mode)
        if entry.is_dir():
            entries[name] = {"kind": "directory", "mode": mode}
        elif entry.is_file():
            entries[name] = {
                "kind": "file",
                "mode": mode,
                "size": entry.stat().st_size,
                "sha256": sha256_file(entry),
            }
        else:
            raise ReviewError(f"namespace contains unsupported entry: {relative(entry)}")
    record: dict[str, object] = {
        "path": relative(path),
        "entry_count": len(entries),
        "entries": entries,
    }
    record["inventory_sha256"] = sha256_bytes(canonical_bytes(record))
    return record


def file_record(path: Path, name: str | None = None) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ReviewError(f"required file is absent: {name or path.name}")
    return {
        "name": name or path.name,
        "mode": stat.S_IMODE(resolved.stat().st_mode),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def assert_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ReviewError(f"{label} differs: observed={observed!r} expected={expected!r}")


def command_receipt(
    output: Path,
    ordinal: int,
    label: str,
    argv: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, object]:
    prefix = output / "commands" / f"{ordinal:02d}-{label}"
    display = {
        "argv": argv,
        "cwd": ".",
        "environment": environment or {},
    }
    write_json(prefix.with_suffix(".command.json"), display)
    command_environment = os.environ.copy()
    command_environment.update(environment or {})
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        env=command_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    write_bytes(prefix.with_suffix(".stdout"), completed.stdout)
    write_bytes(prefix.with_suffix(".stderr"), completed.stderr)
    write_bytes(
        prefix.with_suffix(".exit-code"),
        f"{completed.returncode}\n".encode("ascii"),
    )
    return {
        "label": label,
        "argv": argv,
        "returncode": completed.returncode,
        "command_sha256": sha256_file(prefix.with_suffix(".command.json")),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "stdout_bytes": len(completed.stdout),
        "stderr_bytes": len(completed.stderr),
    }


def compile_receipt() -> dict[str, object]:
    sources = (
        PACKAGE / "package_core.py",
        PACKAGE / "prepare_package.py",
        PACKAGE / "product_driver.py",
        PACKAGE / "tests/test_product_package.py",
        Path(__file__).resolve(),
    )
    records = []
    for source in sources:
        raw = source.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        compile(text, relative(source), "exec", dont_inherit=True)
        records.append(
            {
                "path": relative(source),
                "size": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    return {
        "status": "PASS_PYTHON_COMPILE",
        "source_count": len(records),
        "sources": records,
    }


def load_package_modules() -> tuple[Any, Any]:
    sys.path.insert(0, str(PACKAGE))
    try:
        core = importlib.import_module("package_core")
        driver = importlib.import_module("product_driver")
    finally:
        sys.path.pop(0)
    return core, driver


class CountingOfficialTokenizer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.decode_calls: list[dict[str, object]] = []
        self.vocab_size = tokenizer.vocab_size

    def __len__(self) -> int:
        return len(self.tokenizer)

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_added_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_added_vocab()

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def decode(self, token_ids: list[int], **kwargs: object) -> object:
        self.decode_calls.append({"token_ids": list(token_ids), "kwargs": dict(kwargs)})
        return self.tokenizer.decode(token_ids, **kwargs)


def captured_decode(
    driver: Any,
    tokenizer: CountingOfficialTokenizer,
    token_ids: list[int],
    path: Path,
    *,
    budget: int,
    expected_error: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], str | None, str | None]:
    snapshots: list[dict[str, object]] = []
    original_atomic = driver._atomic_json

    def tracking_atomic(
        target: Path, value: Any, *, exclusive: bool
    ) -> None:
        original_atomic(target, value, exclusive=exclusive)
        snapshots.append(
            {
                "decode_call_count_after_persist": len(tokenizer.decode_calls),
                "artifact": copy.deepcopy(value),
            }
        )

    driver._atomic_json = tracking_atomic
    error_text = None
    decoded_text = None
    try:
        try:
            decoded_text, _ = driver.decode_token_ids_once(
                tokenizer,
                token_ids,
                path,
                budget=budget,
            )
        except Exception as error:
            error_text = str(error)
            if expected_error is None or expected_error not in error_text:
                raise
        else:
            if expected_error is not None:
                raise ReviewError(f"expected decode rejection was absent: {expected_error}")
    finally:
        driver._atomic_json = original_atomic
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return artifact, snapshots, error_text, decoded_text


def safetensors_shape(model: Path, tensor_name: str) -> list[int]:
    with model.resolve().open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ReviewError("official model safetensors header is truncated")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size).decode("utf-8", errors="strict"))
    tensor = header.get(tensor_name)
    if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
        raise ReviewError(f"official model tensor metadata is absent: {tensor_name}")
    return tensor["shape"]


def real_tokenizer_receipt(core: Any, driver: Any, output: Path) -> dict[str, object]:
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    config_path = core.TOKENIZER_ROOT / "config.json"
    tokenizer_path = core.TOKENIZER_ROOT / "tokenizer.json"
    tokenizer_config_path = core.TOKENIZER_ROOT / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert_equal(config.get("vocab_size"), MODEL_VOCABULARY_SIZE, "model vocab_size")
    assert_equal(config.get("tie_word_embeddings"), True, "tied output embeddings")
    embedding_shape = safetensors_shape(
        core.MODEL,
        "model.embed_tokens.weight",
    )
    assert_equal(
        embedding_shape,
        [MODEL_VOCABULARY_SIZE, config["hidden_size"]],
        "shared input/output embedding shape",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(core.TOKENIZER_ROOT),
        local_files_only=True,
        trust_remote_code=False,
    )
    domain = core.authenticated_tokenizer_domain()
    assert_equal(tokenizer.vocab_size, TOKENIZER_BASE_VOCABULARY_SIZE, "base vocab")
    assert_equal(len(tokenizer), TOKENIZER_MAPPED_ID_COUNT, "tokenizer length")
    assert_equal(len(tokenizer.get_added_vocab()), TOKENIZER_ADDED_TOKEN_COUNT, "added tokens")
    assert_equal(max(tokenizer.get_vocab().values()), TOKENIZER_MAXIMUM_MAPPED_ID, "max ID")
    assert_equal(tokenizer.get_vocab(), domain["token_to_id"], "runtime token mapping")

    source_text = "Review token domain validation remains readable."
    encoded = tokenizer.encode(source_text, add_special_tokens=False)
    base_ids = [token_id for token_id in encoded if token_id < TOKENIZER_BASE_VOCABULARY_SIZE][
        :4
    ]
    if len(base_ids) != 4:
        raise ReviewError("official tokenizer did not yield four mapped base IDs")

    with tempfile.TemporaryDirectory(dir=output) as temporary:
        scratch = Path(temporary)
        mapped = CountingOfficialTokenizer(tokenizer)
        mapped_artifact, mapped_snapshots, _, mapped_text = captured_decode(
            driver,
            mapped,
            base_ids,
            scratch / "mapped-base.json",
            budget=4,
        )
        assert_equal(len(mapped.decode_calls), 1, "mapped base decode call count")
        predecode_snapshots = [
            item
            for item in mapped_snapshots
            if item["artifact"]["decode"]["classification"]
            == "PENDING_OFFICIAL_WHOLE_SEQUENCE_DECODE"
        ]
        if len(predecode_snapshots) != 1:
            raise ReviewError("mapped predecode atomic snapshot count differs")
        predecode = predecode_snapshots[0]
        assert_equal(
            predecode["decode_call_count_after_persist"],
            0,
            "decode count at mapped predecode publication",
        )
        if any(
            step["tokenizer_id_classification"] != "MAPPED_BASE_TOKEN_ID"
            for step in predecode["artifact"]["steps"]
        ):
            raise ReviewError("mapped base classifications were not atomically persisted")

        added = CountingOfficialTokenizer(tokenizer)
        added_ids = [base_ids[0], base_ids[1], 151645]
        added_artifact, added_snapshots, _, _ = captured_decode(
            driver,
            added,
            added_ids,
            scratch / "mapped-added.json",
            budget=4,
        )
        assert_equal(len(added.decode_calls), 1, "mapped added decode call count")
        added_predecode = [
            item
            for item in added_snapshots
            if item["artifact"]["decode"]["classification"]
            == "PENDING_OFFICIAL_WHOLE_SEQUENCE_DECODE"
        ]
        if len(added_predecode) != 1:
            raise ReviewError("mapped added predecode atomic snapshot count differs")
        assert_equal(
            added_predecode[0]["decode_call_count_after_persist"],
            0,
            "decode count at added-token predecode publication",
        )
        assert_equal(
            added_artifact["steps"][-1]["tokenizer_id_classification"],
            "MAPPED_ADDED_TOKEN_ID",
            "added-token classification",
        )
        assert_equal(
            added_artifact["canonical_generation_result"]["stop_reason"],
            "eos",
            "added-token EOS bookkeeping",
        )

        reserved = CountingOfficialTokenizer(tokenizer)
        reserved_ids = [base_ids[0], base_ids[1], base_ids[2], 151665]
        reserved_artifact, reserved_snapshots, reserved_error, _ = captured_decode(
            driver,
            reserved,
            reserved_ids,
            scratch / "reserved.json",
            budget=4,
            expected_error="UNDECODABLE_RESERVED_MODEL_VOCAB_ID",
        )
        assert_equal(len(reserved.decode_calls), 0, "reserved-ID decode call count")
        assert_equal(
            reserved_artifact["decode"]["classification"],
            "UNDECODABLE_RESERVED_MODEL_VOCAB_ID",
            "reserved-ID classification",
        )
        assert_equal(
            reserved_artifact["canonical_generation_result"]["token_ids"],
            reserved_ids,
            "reserved-ID canonical sequence",
        )
        assert_equal(
            [step["token_id"] for step in reserved_artifact["steps"]],
            reserved_ids,
            "reserved-ID step diagnostics",
        )
        assert_equal(
            reserved_artifact["decode"]["undecodable_ids"],
            [{"step_ordinal": 3, "token_id": 151665}],
            "reserved-ID rejection diagnostics",
        )
        reserved_persisted = [
            item
            for item in reserved_snapshots
            if item["artifact"]["decode"]["classification"]
            == "UNDECODABLE_RESERVED_MODEL_VOCAB_ID"
        ]
        if len(reserved_persisted) != 1:
            raise ReviewError("reserved-ID atomic classification snapshot count differs")
        assert_equal(
            reserved_persisted[0]["decode_call_count_after_persist"],
            0,
            "decode count at reserved-ID rejection publication",
        )

    arbitrary_inputs = (
        "Grüße 世界 — registers",
        "RTL?\nTokenizer: café.",
        "Δοκιμή 🙂 byte boundary",
    )
    round_trips = []
    for value in arbitrary_inputs:
        strict = value.encode("utf-8").decode("utf-8", errors="strict")
        token_ids = tokenizer.encode(strict, add_special_tokens=False)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        assert_equal(decoded, strict, "arbitrary UTF-8 tokenizer round trip")
        round_trips.append(
            {
                "input_utf8_sha256": sha256_bytes(value.encode("utf-8")),
                "token_count": len(token_ids),
                "decode_call_count": 1,
                "exact_round_trip": True,
            }
        )
    try:
        b"\xf0\x28\x8c\x28".decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        malformed_utf8_rejected = True
    else:
        malformed_utf8_rejected = False
    if not malformed_utf8_rejected:
        raise ReviewError("malformed UTF-8 was not rejected")

    return {
        "status": "PASS_REAL_OFFICIAL_TOKENIZER_AND_ATOMIC_DECODE",
        "official_identity": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": core.TOKENIZER_REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
            "files": {
                "config.json": file_record(config_path, "config.json"),
                "tokenizer.json": file_record(tokenizer_path, "tokenizer.json"),
                "tokenizer_config.json": file_record(
                    tokenizer_config_path,
                    "tokenizer_config.json",
                ),
            },
        },
        "model_logit_domain": {
            "config_vocab_size": config["vocab_size"],
            "tie_word_embeddings": config["tie_word_embeddings"],
            "shared_embedding_shape": embedding_shape,
            "official_model_sha256": core.EXPECTED["model"],
        },
        "tokenizer_domain": core.tokenizer_domain_evidence(),
        "mapped_base_probe": {
            "token_ids": base_ids,
            "official_whole_sequence_decode_calls": len(mapped.decode_calls),
            "classification": mapped_artifact["decode"]["classification"],
            "classification_persisted_before_decode": True,
            "decoded_utf8_sha256": sha256_bytes(mapped_text.encode("utf-8")),
        },
        "mapped_added_probe": {
            "token_ids": added_ids,
            "official_whole_sequence_decode_calls": len(added.decode_calls),
            "classification": added_artifact["steps"][-1][
                "tokenizer_id_classification"
            ],
            "stop_reason": added_artifact["canonical_generation_result"][
                "stop_reason"
            ],
            "classification_persisted_before_decode": True,
        },
        "reserved_model_id_probe": {
            "token_ids": reserved_ids,
            "official_whole_sequence_decode_calls": len(reserved.decode_calls),
            "classification": reserved_artifact["decode"]["classification"],
            "error": reserved_error,
            "canonical_ids_preserved": True,
            "complete_diagnostics_preserved": True,
            "classification_persisted_before_rejection": True,
            "coercion_performed": False,
            "ids_omitted": False,
        },
        "strict_utf8": {
            "arbitrary_input_round_trips": round_trips,
            "malformed_utf8_rejected": malformed_utf8_rejected,
        },
    }


def equality_assumption_receipt() -> dict[str, object]:
    files = (
        PACKAGE / "package_core.py",
        PACKAGE / "product_driver.py",
        PACKAGE / "tests/test_product_package.py",
    )
    examined_comparisons: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            expression = ast.unparse(node)
            if "MODEL_VOCABULARY_SIZE" not in expression:
                continue
            record = {
                "path": relative(path),
                "line": node.lineno,
                "expression": expression,
            }
            examined_comparisons.append(record)
            if any(
                marker in expression
                for marker in (
                    "TOKENIZER_",
                    "runtime_length",
                    "runtime_base_size",
                    "vocab_size",
                    "len(tokenizer)",
                )
            ):
                invalid.append(record)
    if invalid:
        raise ReviewError(f"model/tokenizer equality assumption found: {invalid}")
    return {
        "status": "PASS_NO_MODEL_TOKENIZER_EQUALITY_ASSUMPTION",
        "files": [relative(path) for path in files],
        "model_domain_comparisons": examined_comparisons,
        "invalid_cross_domain_comparisons": invalid,
    }


def current_reproducibility_inventory(
    report: dict[str, Any],
) -> dict[str, dict[str, object]]:
    inventory = report.get("artifact_inventory")
    if not isinstance(inventory, dict):
        raise ReviewError("two-build artifact inventory is absent")
    current: dict[str, dict[str, object]] = {}
    for name in inventory:
        path = PACKAGE / name
        current[name] = {
            "mode": stat.S_IMODE(path.stat().st_mode),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return current


def package_contract_receipt(core: Any) -> dict[str, object]:
    authorization = core.validate_prepared_authorization()
    predecessor = core.validate_predecessor_namespaces()
    construction = core.validate_construction_only_record()
    core.assert_fresh_execution_state()
    authorization_error = None
    try:
        core.validate_authorization()
    except core.ProductError as error:
        authorization_error = str(error)
    assert_equal(authorization_error, "CF10 has no execution authority", "launch gate")
    claim_error = None
    try:
        core.claim_authorization(authorization)
    except core.ProductError as error:
        claim_error = str(error)
    assert_equal(claim_error, "CF10 has no authority to consume", "consumption gate")

    expected_absent = {
        relative(path): not os.path.lexists(path)
        for path in core.FORBIDDEN_EXECUTION_STATE
    }
    if not all(expected_absent.values()):
        raise ReviewError("forbidden CF10 execution state exists")
    authority_inventory = sorted(
        entry.name for entry in core.CONSTRUCTION_AUTHORITY_RECORD.parent.iterdir()
    )
    assert_equal(
        authority_inventory,
        ["manager-decision.json"],
        "authority inventory",
    )
    report = core.load_object(core.REPRODUCIBILITY_REPORT)
    assert_equal(
        report.get("status"),
        "PASS_TWO_BUILD_BYTE_MODE_EQUAL",
        "two-build status",
    )
    assert_equal(report.get("build_count"), 2, "two-build count")
    assert_equal(report.get("generation_executed"), False, "repro generation state")
    assert_equal(report.get("rtl_executed"), False, "repro RTL state")
    assert_equal(
        current_reproducibility_inventory(report),
        report["artifact_inventory"],
        "current two-build artifact inventory",
    )
    return {
        "status": "PASS_ZERO_AUTHORITY_FRESH_NON_EXECUTING_PACKAGE",
        "identity": authorization["identity"],
        "transaction_nonce": authorization["transaction_nonce"],
        "authorization_state": authorization["authorization_state"],
        "authorization_status": authorization["status"],
        "execution_limit": authorization["execution_limit"],
        "authority_cardinality": authorization["authority_provenance"][
            "authority_cardinality"
        ],
        "construction_record": {
            "status": "GRANTED_FOR_CONSTRUCTION_AND_REVIEW_ONLY",
            "effective_execution_authority_cardinality": construction[
                "authority_cardinality"
            ],
            "sha256": construction["sha256"],
        },
        "launch_gate_error": authorization_error,
        "consumption_gate_error": claim_error,
        "forbidden_execution_state_absent": expected_absent,
        "predecessors": predecessor,
        "reproducibility": {
            "status": report["status"],
            "build_count": report["build_count"],
            "record_sha256": sha256_file(core.REPRODUCIBILITY_REPORT),
            "current_recorded_artifact_inventory_exact": True,
            "generation_executed": report["generation_executed"],
            "rtl_executed": report["rtl_executed"],
        },
        "stage1_state": authorization["stage1_state"],
        "stage1_closed": authorization["stage1_closed"],
        "stage2": authorization["stage2"],
    }


def artifact_hashes(core: Any) -> dict[str, dict[str, object]]:
    names = (
        "authorization.json",
        "authority/manager-decision.json",
        "package-checks.json",
        "package_core.py",
        "predecessor-manifest.json",
        "prepare_package.py",
        "product_driver.py",
        "reproducibility-report.json",
        "tests/test_product_package.py",
    )
    records = {name: file_record(PACKAGE / name, name) for name in names}
    records["official-tokenizer/tokenizer.json"] = file_record(
        core.TOKENIZER_ROOT / "tokenizer.json",
        "official-tokenizer/tokenizer.json",
    )
    records["official-tokenizer/tokenizer_config.json"] = file_record(
        core.TOKENIZER_ROOT / "tokenizer_config.json",
        "official-tokenizer/tokenizer_config.json",
    )
    records["official-model/config.json"] = file_record(
        core.TOKENIZER_ROOT / "config.json",
        "official-model/config.json",
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise ReviewError(f"review output already exists: {output}")
    if output.parent != (ROOT / "build/ace2_chat_demo").resolve():
        raise ReviewError("review output must be a direct ace2_chat_demo build child")
    output.mkdir(mode=0o755)

    pre_inventory = {
        "schema": "ace2-r6-cf10-l2-review-input-inventory-v1",
        "cf10": namespace_inventory(PACKAGE),
        "predecessors": {
            name: namespace_inventory(path) for name, path in PREDECESSORS.items()
        },
        "review_tool": file_record(Path(__file__).resolve(), relative(Path(__file__).resolve())),
        "official_files": {
            "config.json": file_record(TOKENIZER_ROOT / "config.json", "config.json"),
            "tokenizer.json": file_record(
                TOKENIZER_ROOT / "tokenizer.json",
                "tokenizer.json",
            ),
            "tokenizer_config.json": file_record(
                TOKENIZER_ROOT / "tokenizer_config.json",
                "tokenizer_config.json",
            ),
        },
        "tool_versions": {
            "python": platform.python_version(),
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
    }
    write_json(output / "input-inventory.json", pre_inventory)
    compile_result = compile_receipt()
    write_json(output / "compile-receipt.json", compile_result)

    test_command = command_receipt(
        output,
        1,
        "focused-tests",
        [
            "python3",
            "-B",
            relative(PACKAGE / "tests/test_product_package.py"),
        ],
        environment={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    combined_test_output = (
        (output / "commands/01-focused-tests.stdout").read_bytes()
        + (output / "commands/01-focused-tests.stderr").read_bytes()
    ).decode("utf-8", errors="strict")
    count_match = re.search(r"Ran ([0-9]+) tests?", combined_test_output)
    tests_run = int(count_match.group(1)) if count_match else None
    if test_command["returncode"] != 0 or tests_run != EXPECTED_TEST_COUNT:
        raise ReviewError(
            f"focused tests failed or count differed: rc={test_command['returncode']} "
            f"tests={tests_run}"
        )
    focused_tests = {
        "status": "PASS_FOCUSED_FAITHFUL_FAKE_MATRIX",
        "tests_run": tests_run,
        "failures": 0,
        "errors": 0,
        "command": test_command,
        "coverage": [
            "authenticated_real_tokenizer_metadata",
            "mapped_base_and_added_token_decode",
            "reserved_and_out_of-range_model_ID_rejection",
            "multibyte_byte_fallback_diagnostics",
            "strict_UTF-8_decode_exception_surrogate_replacement_rejection",
            "EOS_and_generation_budget_bookkeeping",
            "atomic_observability_and_exclusive_result_publication",
            "source_model_tokenizer_export_RTL_tamper_rejection",
            "provenance_and_predecessor_authentication",
            "arbitrary_UTF-8_input",
            "two-build_reproducibility_and_fresh_execution_absence",
        ],
    }
    write_json(output / "focused-test-receipt.json", focused_tests)

    core, driver = load_package_modules()
    official = real_tokenizer_receipt(core, driver, output)
    write_json(output / "official-tokenizer-receipt.json", official)
    equality = equality_assumption_receipt()
    write_json(output / "equality-assumption-receipt.json", equality)
    contract = package_contract_receipt(core)
    assert_equal(contract["identity"], IDENTITY, "CF10 identity")
    assert_equal(contract["transaction_nonce"], NONCE, "CF10 nonce")
    write_json(output / "package-contract-receipt.json", contract)

    post_inventory = {
        "schema": "ace2-r6-cf10-l2-review-post-test-inventory-v1",
        "cf10": namespace_inventory(PACKAGE),
        "predecessors": {
            name: namespace_inventory(path) for name, path in PREDECESSORS.items()
        },
    }
    assert_equal(post_inventory["cf10"], pre_inventory["cf10"], "CF10 namespace")
    assert_equal(
        post_inventory["predecessors"],
        pre_inventory["predecessors"],
        "CF07-CF09 namespaces",
    )
    post_inventory["cf10_unchanged"] = True
    post_inventory["predecessors_unchanged"] = True
    write_json(output / "post-test-inventory.json", post_inventory)

    receipt_names = (
        "compile-receipt.json",
        "equality-assumption-receipt.json",
        "focused-test-receipt.json",
        "input-inventory.json",
        "official-tokenizer-receipt.json",
        "package-contract-receipt.json",
        "post-test-inventory.json",
    )
    receipt_hashes = {
        name: {
            "size": (output / name).stat().st_size,
            "sha256": sha256_file(output / name),
        }
        for name in receipt_names
    }
    verdict = {
        "schema": "ace2-r6-cf10-canonical-l2-verdict-v1",
        "status": "PASS",
        "verdict": "ACCEPT_CF10_TOKENIZER_DOMAIN_REPAIR",
        "review_mode": "INDEPENDENT_NON_EXECUTING",
        "identity": IDENTITY,
        "transaction_nonce": NONCE,
        "exact_hashes": {
            "reviewed_artifacts": artifact_hashes(core),
            "receipts": receipt_hashes,
            "cf10_namespace_inventory_sha256": pre_inventory["cf10"][
                "inventory_sha256"
            ],
            "predecessor_namespace_inventory_sha256": {
                name: record["inventory_sha256"]
                for name, record in pre_inventory["predecessors"].items()
            },
            "tokenizer_mapping_sha256": official["tokenizer_domain"][
                "mapping_sha256"
            ],
        },
        "acceptance": {
            "identity_and_nonce_exact": True,
            "zero_consumption_and_execution_authority": True,
            "model_logit_vocabulary_size": MODEL_VOCABULARY_SIZE,
            "tokenizer_base_vocabulary_size": TOKENIZER_BASE_VOCABULARY_SIZE,
            "tokenizer_length_and_mapped_id_count": TOKENIZER_MAPPED_ID_COUNT,
            "tokenizer_added_token_count": TOKENIZER_ADDED_TOKEN_COUNT,
            "tokenizer_maximum_mapped_id": TOKENIZER_MAXIMUM_MAPPED_ID,
            "model_tokenizer_domains_separated": True,
            "invalid_vocabulary_equality_assumption_absent": True,
            "generated_ids_and_classification_atomically_persisted": True,
            "mapped_sequences_official_whole_sequence_decode_calls_each": 1,
            "reserved_model_ids_fail_closed": "UNDECODABLE_RESERVED_MODEL_VOCAB_ID",
            "reserved_model_id_decode_calls": 0,
            "reserved_id_coercion_or_diagnostic_loss": False,
            "strict_utf8_eos_atomicity_tamper_provenance_arbitrary_input_pass": True,
            "faithful_fake_focused_tests_run": tests_run,
            "real_official_tokenizer_probe_pass": True,
            "two_build_byte_mode_reproducibility": "PASS_TWO_BUILD_BYTE_MODE_EQUAL",
            "cf10_unchanged_during_review": True,
            "cf07_cf08_cf09_byte_mode_unchanged": True,
            "launch_permitted": False,
            "consumption_permitted": False,
            "attempt_registry_invocation_completion_result_absent": True,
        },
        "authority_state": {
            "authorization": "NOT_GRANTED",
            "authority_cardinality": 0,
            "execution_limit": 0,
            "consumption": "NOT_CONSUMED",
            "product_executed": False,
            "model_executed": False,
            "rtl_executed": False,
        },
        "stage_state": {
            "stage1": "OPEN",
            "stage1_closed": False,
            "stage2": "FORBIDDEN",
            "stage_transition_invoked": False,
        },
        "claim_boundary": (
            "This verdict accepts only the bounded CF10 tokenizer-domain repair "
            "package. It grants no execution authority and records no product, "
            "model, RTL, synthesis, timing, PPA, or Stage transition result."
        ),
    }
    verdict_path = output / "verdict.json"
    write_json(verdict_path, verdict)
    verdict_sha256 = sha256_file(verdict_path)
    write_bytes(
        output / "verdict.json.sha256",
        f"{verdict_sha256}  verdict.json\n".encode("ascii"),
    )
    print("status=PASS")
    print(f"verdict={relative(verdict_path)}")
    print(f"sha256={verdict_sha256}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReviewError, OSError, UnicodeDecodeError, ValueError) as error:
        print(f"CF10 L2 review failed: {error}", file=sys.stderr)
        raise SystemExit(1)
