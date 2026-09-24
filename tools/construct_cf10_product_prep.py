#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf09"
PACKAGE = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf10"
TERMINAL_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6stage1terminalseal09"
)
TERMINAL_SEAL = SOURCE / "terminal-seal.json"
IDENTITY = "stage1-w4a8-r6-product-prep-cf10-attempt-0001"
NONCE = "5e6c6eb08db41c414a58d73fba75308b8fee26f2c37c7c1624048b2cc57ce44b"
CONSTRUCTION_DECISION = "mgr-r6-cf10-tokenizer-domain-package-01"
PREDECESSORS = {
    "cf07": ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf07",
    "cf08": ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf08",
    "cf09": SOURCE,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new)


def replace_region(
    text: str, start: str, end: str, replacement: str, label: str
) -> str:
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if start_index < 0 or end_index < 0:
        raise RuntimeError(f"{label}: region markers are absent")
    return text[:start_index] + replacement + text[end_index:]


def namespace_manifest(path: Path) -> dict[str, object]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"predecessor namespace is absent or linked: {path}")
    entries: dict[str, dict[str, object]] = {}
    for entry in sorted(path.rglob("*")):
        relative = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            raise RuntimeError(f"predecessor namespace contains a link: {entry}")
        if entry.is_dir():
            entries[relative] = {
                "kind": "directory",
                "mode": entry.stat().st_mode & 0o777,
            }
        elif entry.is_file():
            entries[relative] = {
                "kind": "file",
                "mode": entry.stat().st_mode & 0o777,
                "size": entry.stat().st_size,
                "sha256": sha256(entry),
            }
        else:
            raise RuntimeError(f"predecessor namespace has unsupported entry: {entry}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "entry_count": len(entries),
        "entries": entries,
    }


def construction_record() -> dict[str, object]:
    return {
        "schema": "ace2-r6-external-manager-authority-v1",
        "status": "GRANTED",
        "authority_cardinality": 1,
        "semantic_fields": {
            "decision_id": CONSTRUCTION_DECISION,
            "scope": "r6tokenizervocabrepair10",
            "authorized_action": "CONSTRUCTION_AND_INDEPENDENT_REVIEW_ONLY",
            "sealed_cf07_identity": (
                "stage1-w4a8-r6-product-prep-cf07-attempt-0001"
            ),
            "sealed_cf08_identity": (
                "stage1-w4a8-r6-product-prep-cf08-attempt-0001"
            ),
            "sealed_cf09_identity": (
                "stage1-w4a8-r6-product-prep-cf09-attempt-0001"
            ),
            "sealed_cf07_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "sealed_cf08_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "sealed_cf09_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "sealed_cf09_outcome": "SEALED_RUNTIME_FAILURE",
            "identity": IDENTITY,
            "transaction_nonce": NONCE,
            "execution_limit": 0,
            "retry": "FORBIDDEN",
            "replay": "FORBIDDEN",
            "resume": "FORBIDDEN",
            "relaunch": "FORBIDDEN",
            "software_fallback": False,
            "argv": [
                str(PACKAGE / "launch.sh"),
                "--input-file",
                str(PACKAGE / "input.txt"),
                "--max-new-tokens",
                "4",
            ],
            "execution_in_this_task": "FORBIDDEN",
            "product_execution": "FORBIDDEN",
            "endpoint_execution": "FORBIDDEN",
            "model_execution": "FORBIDDEN",
            "rtl_execution": "FORBIDDEN",
            "independent_l2_review": "REQUIRED",
            "stage1_state": "OPEN",
            "stage2": "FORBIDDEN",
            "stage_transition": "DISABLED",
        },
    }


def make_core() -> str:
    text = (SOURCE / "package_core.py").read_text(encoding="utf-8")
    text = text.replace(
        '"stage1-w4a8-r6-product-prep-cf09-attempt-0001"', f'"{IDENTITY}"'
    )
    text = text.replace(
        '"1f4f1754cc8bce90e836c29bdd57e03b9e5e927244d76cad97bf039e92abcb7f"',
        f'"{NONCE}"',
    )
    text = text.replace(
        '"r6-final-export-product-cf09"', '"r6-final-export-product-cf10"'
    )
    text = replace_region(
        text,
        "AUTHORITY_MISSION = Path(",
        "CF08_TERMINAL_LATEST =",
        "",
        "obsolete execution-authority constants",
    )
    text = replace_region(
        text,
        "PRE_AUTHORITY_HASHES = {",
        "GENERATION_BUDGET = 4",
        "",
        "obsolete pre-authority hashes",
    )
    text = replace_once(
        text,
        "VOCABULARY_SIZE = 151936\n",
        """MODEL_VOCABULARY_SIZE = 151936
TOKENIZER_BASE_VOCABULARY_SIZE = 151643
TOKENIZER_MAPPED_ID_COUNT = 151665
TOKENIZER_ADDED_TOKEN_COUNT = 22
TOKENIZER_MAX_MAPPED_ID = 151664
""",
        "vocabulary domain constants",
    )
    terminal_constants = f'''
CF09_TERMINAL_LATEST = Path({str(TERMINAL_HANDOFF / "latest.json")!r})
CF09_TERMINAL_SEAL = Path({str(TERMINAL_SEAL)!r})
CF09_TERMINAL_LATEST_SHA256 = {sha256(TERMINAL_HANDOFF / "latest.json")!r}
CF09_TERMINAL_HANDOFF_SHA256 = {sha256(TERMINAL_HANDOFF / "round-0001.json")!r}
CF09_TERMINAL_SEAL_SHA256 = {sha256(TERMINAL_SEAL)!r}
CF09_TERMINAL_MISSION_ID = "r6stage1terminalseal09"
PREDECESSOR_MANIFEST = PACKAGE / "predecessor-manifest.json"
PREDECESSOR_NAMESPACES = {{
    "cf07": ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf07",
    "cf08": ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf08",
    "cf09": ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf09",
}}
'''
    text = replace_once(
        text,
        'CF08_TERMINAL_MISSION_ID = "r6stage1terminalseal08"\n',
        'CF08_TERMINAL_MISSION_ID = "r6stage1terminalseal08"\n' + terminal_constants,
        "CF09 terminal constants",
    )
    text = text.replace(
        '    "authority/manager-execution-authority.json",\n', ""
    )
    text = replace_once(
        text,
        '    "input.txt",\n',
        '    "input.txt",\n    "predecessor-manifest.json",\n',
        "predecessor source binding",
    )
    authority_region = '''def construction_only_decision_record() -> dict[str, Any]:
    return {
        "schema": "ace2-r6-external-manager-authority-v1",
        "status": "GRANTED",
        "authority_cardinality": 1,
        "semantic_fields": {
            "decision_id": "mgr-r6-cf10-tokenizer-domain-package-01",
            "scope": "r6tokenizervocabrepair10",
            "authorized_action": "CONSTRUCTION_AND_INDEPENDENT_REVIEW_ONLY",
            "sealed_cf07_identity": (
                "stage1-w4a8-r6-product-prep-cf07-attempt-0001"
            ),
            "sealed_cf08_identity": (
                "stage1-w4a8-r6-product-prep-cf08-attempt-0001"
            ),
            "sealed_cf09_identity": (
                "stage1-w4a8-r6-product-prep-cf09-attempt-0001"
            ),
            "sealed_cf07_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "sealed_cf08_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "sealed_cf09_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "sealed_cf09_outcome": "SEALED_RUNTIME_FAILURE",
            "identity": ONE_USE_IDENTITY,
            "transaction_nonce": TRANSACTION_NONCE,
            "execution_limit": 0,
            "retry": "FORBIDDEN",
            "replay": "FORBIDDEN",
            "resume": "FORBIDDEN",
            "relaunch": "FORBIDDEN",
            "software_fallback": False,
            "argv": authorized_launch_argv(),
            "execution_in_this_task": "FORBIDDEN",
            "product_execution": "FORBIDDEN",
            "endpoint_execution": "FORBIDDEN",
            "model_execution": "FORBIDDEN",
            "rtl_execution": "FORBIDDEN",
            "independent_l2_review": "REQUIRED",
            "stage1_state": "OPEN",
            "stage2": "FORBIDDEN",
            "stage_transition": "DISABLED",
        },
    }


def validate_construction_only_record() -> dict[str, Any]:
    authority_directory = CONSTRUCTION_AUTHORITY_RECORD.parent
    if (
        not authority_directory.is_dir()
        or authority_directory.is_symlink()
        or {entry.name for entry in authority_directory.iterdir()}
        != {CONSTRUCTION_AUTHORITY_RECORD.name}
    ):
        raise ProductError("construction authority inventory differs")
    evidence = _validate_immutable_authority_record(
        CONSTRUCTION_AUTHORITY_RECORD,
        construction_only_decision_record(),
        "construction-only Manager record",
    )
    return {
        **evidence,
        "authorization_state": "REJECTED_CONSTRUCTION_ONLY",
        "authority_cardinality": 0,
    }


def manager_decision_record() -> dict[str, Any]:
    raise ProductError("CF10 has no execution authority")


def validate_manager_authority() -> dict[str, Any]:
    raise ProductError("CF10 has no execution authority")


def _validate_immutable_authority_record(
    path: Path, expected: dict[str, Any], label: str
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProductError(f"{label} is absent or linked")
    record = load_object(path)
    if canonical_bytes(record) != canonical_bytes(expected):
        raise ProductError(f"{label} semantic fields differ")
    if path.read_bytes() != canonical_bytes(record):
        raise ProductError(f"{label} is not canonical JSON")
    if path.stat().st_mode & 0o222:
        raise ProductError(f"{label} is mutable")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "semantic_fields": expected["semantic_fields"],
    }


def validate_cf09_terminal_failure() -> dict[str, Any]:
    require_file(
        CF09_TERMINAL_LATEST,
        CF09_TERMINAL_LATEST_SHA256,
        "CF09 terminal latest handoff",
    )
    review = validate_reviewer_gate(
        CF09_TERMINAL_LATEST, CF09_TERMINAL_MISSION_ID
    )
    if review["handoff_sha256"] != CF09_TERMINAL_HANDOFF_SHA256:
        raise ProductError("CF09 terminal reviewed handoff differs")
    require_file(
        CF09_TERMINAL_SEAL,
        CF09_TERMINAL_SEAL_SHA256,
        "CF09 terminal seal",
    )
    seal = load_object(CF09_TERMINAL_SEAL)
    identity = seal.get("identity", {})
    terminal = seal.get("terminal", {})
    authorization = seal.get("authorization", {})
    artifacts = seal.get("execution_artifacts", {})
    root_cause = seal.get("root_cause", {})
    tokenizer = root_cause.get("authenticated_official_auto_tokenizer", {})
    padded = root_cause.get("model_rows_without_tokenizer_mapping", {})
    if (
        seal.get("status") != "SEALED_RUNTIME_FAILURE"
        or seal.get("first_natural_terminal") is not True
        or seal.get("durable_process_starts") != 1
        or identity.get("package_attempt")
        != "stage1-w4a8-r6-product-prep-cf09-attempt-0001"
        or terminal.get("exit_code") != 2
        or terminal.get("stderr")
        != "product execution refused: official tokenizer vocabulary bounds differ\\n"
        or terminal.get("stdout") != ""
        or terminal.get("timeout_triggered") is not False
        or terminal.get("orphaned") is not False
        or authorization.get("consumption_status")
        != "CONSUMED_BEFORE_PRODUCT_EXECUTION"
        or any(
            authorization.get(action) != "PERMANENTLY_FORBIDDEN"
            for action in ("retry", "replay", "resume", "relaunch")
        )
        or artifacts.get("attempt_0001", {}).get("status") != "ABSENT"
        or artifacts.get("predecode_observability", {}).get("status") != "ABSENT"
        or artifacts.get("execution_registry", {}).get("status") != "ABSENT"
        or artifacts.get("endpoint_invocation", {}).get("status") != "ABSENT"
        or artifacts.get("completion_directory", {}).get("status") != "ABSENT"
        or artifacts.get("completion_summary", {}).get("status") != "ABSENT"
        or artifacts.get("product_result", {}).get("status") != "ABSENT"
        or root_cause.get("classification")
        != "TOKENIZER_MODEL_VOCABULARY_DOMAIN_CONTRACT_DEFECT"
        or root_cause.get("model_output_logit_dimension") != 151936
        or tokenizer.get("base_vocab_size") != 151643
        or tokenizer.get("added_token_count") != 22
        or tokenizer.get("tokenizer_length") != 151665
        or tokenizer.get("maximum_mapped_token_id") != 151664
        or padded != {"first": 151665, "last": 151935, "count": 271}
    ):
        raise ProductError("CF09 terminal failure semantics differ")
    return {
        **review,
        "terminal_seal": str(CF09_TERMINAL_SEAL),
        "terminal_seal_sha256": CF09_TERMINAL_SEAL_SHA256,
        "status": "AUTHENTICATED_TOKENIZER_DOMAIN_CONTRACT_DEFECT",
        "model_vocabulary_size": 151936,
        "tokenizer_base_vocabulary_size": 151643,
        "tokenizer_mapped_id_count": 151665,
        "tokenizer_added_token_count": 22,
        "tokenizer_maximum_mapped_id": 151664,
        "authority_consumed": True,
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "generated_token_ids": "NOT_PRESERVED",
        "product_result": "ABSENT",
    }


'''
    text = replace_region(
        text,
        "def construction_only_decision_record()",
        "def validate_execution_argv(",
        authority_region,
        "construction authority and CF09 terminal gate",
    )
    old_gates = '''        "qualification_binding": verify_qualification_bindings(),
        "cf08_terminal_failure": validate_cf08_terminal_failure(),
        "manager_execution_authority": validate_manager_authority(),
        "construction_only_record": validate_construction_only_record(),
'''
    new_gates = '''        "qualification_binding": verify_qualification_bindings(),
        "cf09_terminal_failure": validate_cf09_terminal_failure(),
        "tokenizer_domain": tokenizer_domain_evidence(),
        "predecessor_namespaces": validate_predecessor_namespaces(),
        "construction_only_record": validate_construction_only_record(),
'''
    text = replace_once(text, old_gates, new_gates, "prerequisite gates")
    claims_region = '''def commit_authorization_claim(path: Path = CONSUMED) -> None:
    raise ProductError("CF10 has no authority to consume")


def claim_authorization(
    authorization: dict[str, Any], path: Path = CONSUMED
) -> None:
    raise ProductError("CF10 has no authority to consume")


'''
    text = replace_region(
        text,
        "def commit_authorization_claim(",
        "AUTHORIZATION_KEYS =",
        claims_region,
        "consumption functions",
    )
    text = text.replace('    "manager_execution_authority_sha256",\n', "")
    text = text.replace('    "reviewed_pre_authority_package",\n', "")
    text = replace_once(
        text,
        '    "tokenizer_json_sha256",\n',
        '    "tokenizer_json_sha256",\n    "tokenizer_domain",\n',
        "tokenizer-domain binding key",
    )
    validation_region = '''def validate_authorization_document(authorization: dict[str, Any]) -> None:
    construction = validate_construction_only_record()
    expected_provenance = {
        "origin": "CONSTRUCTION_ONLY_MANAGER_DECISION",
        "authority_cardinality": 0,
        "manager_decision": construction,
        "execution_authority": "NOT_GRANTED",
        "independent_review": "REQUIRED_BEFORE_ANY_LATER_EXECUTION_AUTHORITY",
    }
    if set(authorization) != AUTHORIZATION_KEYS:
        raise ProductError("literal product authorization sidecar field set differs")
    if (
        authorization.get("schema")
        != "ace2-r6-stage1-product-authorization-sidecar-v1"
        or authorization.get("status")
        != "PREPARED_NOT_AUTHORIZED_NOT_CONSUMED"
        or authorization.get("authorization_state") != "NOT_GRANTED"
        or authorization.get("authority_provenance") != expected_provenance
        or authorization.get("execution_limit") != 0
        or authorization.get("retry") != "FORBIDDEN"
        or authorization.get("replay") != "FORBIDDEN"
        or authorization.get("resume") != "FORBIDDEN"
        or authorization.get("relaunch") != "FORBIDDEN"
        or authorization.get("identity") != ONE_USE_IDENTITY
        or authorization.get("generation_budget") != GENERATION_BUDGET
        or authorization.get("decoding")
        != "official-tokenizer-whole-sequence-exactly-once"
        or authorization.get("transaction_nonce") != TRANSACTION_NONCE
        or authorization.get("completion_directory") != str(COMPLETION)
        or authorization.get("argv") != authorized_launch_argv()
        or authorization.get("endpoint_argv") != authorized_endpoint_argv()
        or authorization.get("input_utf8_path") != str(PACKAGE / "input.txt")
        or authorization.get("accelerator_agreement") != "NOT_EXECUTED"
        or authorization.get("readable_dialogue") != "NOT_EXECUTED"
        or authorization.get("latency") != "NOT_MEASURED"
        or authorization.get("latency_scopes")
        != {
            "accelerator_latency": "NOT_MEASURED",
            "endpoint_process_wall": "NOT_MEASURED",
            "host_model_orchestration": "NOT_MEASURED",
            "simulator_cycles": "NOT_EXECUTED",
        }
        or authorization.get("software_fallback") is not False
        or authorization.get("stage1_state") != "OPEN"
        or authorization.get("stage1_closed") is not False
        or authorization.get("stage2") != "FORBIDDEN"
    ):
        raise ProductError("literal product authorization sidecar fields differ")
    bindings = authorization.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != BINDING_KEYS:
        raise ProductError("authorization binding field set differs")


def validate_binding_snapshot(bindings: dict[str, Any]) -> None:
    expected = {
        "checkpoint_sha256": EXPECTED["checkpoint"],
        "export_manifest_sha256": EXPECTED["export_manifest"],
        "packed_w4_sha256": EXPECTED["packed_w4"],
        "weight_scale32_sha256": EXPECTED["weight_scale32"],
        "ordered_rows_sha256": EXPECTED["ordered_rows"],
        "per_row_digest_ledger_sha256": EXPECTED["row_digests"],
        "tokenizer_json_sha256": EXPECTED["tokenizer_json"],
        "tokenizer_config_sha256": EXPECTED["tokenizer_config"],
        "tokenizer_domain": tokenizer_domain_evidence(),
        "official_model_sha256": EXPECTED["model"],
        "rtl_source_closure": rtl_source_closure(),
        "public_ace2_shell_interface": public_ace2_shell_interface(),
    }
    for field, value in expected.items():
        if bindings.get(field) != value:
            raise ProductError(f"authorization literal binding differs: {field}")


'''
    text = replace_region(
        text,
        "def validate_authorization_document(",
        "def assert_fresh_execution_state(",
        validation_region,
        "zero-authority authorization validation",
    )
    text = text.replace(
        'raise ProductError(f"CF09 execution state is not fresh: {present}")',
        'raise ProductError(f"CF10 execution state is not fresh: {present}")',
    )
    closure_region = '''def validate_authority_closure_documents(
    bindings: dict[str, Any]
) -> None:
    checks = load_object(PACKAGE / "package-checks.json")
    check_values = checks.get("checks", {})
    report = load_object(REPRODUCIBILITY_REPORT)
    if (
        checks.get("status") != "PASS_NON_EXECUTING"
        or check_values.get("external_authorization_granted") is not False
        or check_values.get("authority_cardinality") != 0
        or check_values.get("execution_review_completed") is not False
        or check_values.get("execution_permitted_in_this_task") is not False
        or check_values.get("generation_executed") is not False
        or check_values.get("rtl_executed") is not False
        or report.get("status") != "PASS_TWO_BUILD_BYTE_MODE_EQUAL"
        or report.get("build_count") != 2
        or report.get("generation_executed") is not False
        or report.get("rtl_executed") is not False
        or report.get("stage1_state") != "OPEN"
        or report.get("stage2") != "FORBIDDEN"
    ):
        raise ProductError("construction-only reproducibility closure differs")


'''
    text = replace_region(
        text,
        "def validate_authority_closure_documents(",
        "def validate_prepared_authorization(",
        closure_region,
        "construction-only closure validation",
    )
    text = replace_region(
        text,
        "def validate_authorization()",
        "def host_time()",
        '''def validate_authorization() -> dict[str, Any]:
    raise ProductError("CF10 has no execution authority")


def validate_execution_review() -> dict[str, Any]:
    raise ProductError("CF10 execution authority is a separate later mission")


''',
        "execution authorization gate",
    )
    metadata_helpers = '''def authenticated_tokenizer_domain() -> dict[str, Any]:
    tokenizer_path = (TOKENIZER_ROOT / "tokenizer.json").resolve()
    require_file(
        tokenizer_path,
        EXPECTED["tokenizer_json"],
        "official tokenizer metadata",
    )
    document = load_object(tokenizer_path)
    model = document.get("model")
    base_vocab = model.get("vocab") if isinstance(model, dict) else None
    added_tokens = document.get("added_tokens")
    if not isinstance(base_vocab, dict) or not isinstance(added_tokens, list):
        raise ProductError("official tokenizer vocabulary metadata is malformed")
    if (
        len(base_vocab) != TOKENIZER_BASE_VOCABULARY_SIZE
        or set(base_vocab.values()) != set(range(TOKENIZER_BASE_VOCABULARY_SIZE))
    ):
        raise ProductError("official tokenizer base vocabulary domain differs")
    added_token_to_id: dict[str, int] = {}
    for entry in added_tokens:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("content"), str)
            or type(entry.get("id")) is not int
            or entry["content"] in added_token_to_id
            or entry["content"] in base_vocab
        ):
            raise ProductError("official added-token mapping is malformed")
        added_token_to_id[entry["content"]] = entry["id"]
    token_to_id = {**base_vocab, **added_token_to_id}
    mapped_ids = frozenset(token_to_id.values())
    added_ids = frozenset(added_token_to_id.values())
    if (
        len(added_ids) != TOKENIZER_ADDED_TOKEN_COUNT
        or len(mapped_ids) != TOKENIZER_MAPPED_ID_COUNT
        or max(mapped_ids, default=-1) != TOKENIZER_MAX_MAPPED_ID
        or mapped_ids != frozenset(range(TOKENIZER_MAPPED_ID_COUNT))
        or not added_ids.isdisjoint(
            frozenset(range(TOKENIZER_BASE_VOCABULARY_SIZE))
        )
    ):
        raise ProductError("official tokenizer mapped decode domain differs")
    mapping_sha256 = sha256_bytes(
        canonical_bytes(sorted(token_to_id.items(), key=lambda item: item[0]))
    )
    return {
        "model_vocabulary_size": MODEL_VOCABULARY_SIZE,
        "tokenizer_base_vocabulary_size": TOKENIZER_BASE_VOCABULARY_SIZE,
        "tokenizer_mapped_id_count": TOKENIZER_MAPPED_ID_COUNT,
        "tokenizer_added_token_count": TOKENIZER_ADDED_TOKEN_COUNT,
        "tokenizer_maximum_mapped_id": TOKENIZER_MAX_MAPPED_ID,
        "mapping_sha256": mapping_sha256,
        "mapped_ids": mapped_ids,
        "added_ids": added_ids,
        "token_to_id": token_to_id,
        "added_token_to_id": added_token_to_id,
    }


def tokenizer_domain_evidence() -> dict[str, Any]:
    domain = authenticated_tokenizer_domain()
    return {
        key: domain[key]
        for key in (
            "model_vocabulary_size",
            "tokenizer_base_vocabulary_size",
            "tokenizer_mapped_id_count",
            "tokenizer_added_token_count",
            "tokenizer_maximum_mapped_id",
            "mapping_sha256",
        )
    }


def _namespace_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise ProductError(f"predecessor namespace is absent or linked: {path}")
    entries: dict[str, dict[str, Any]] = {}
    for entry in sorted(path.rglob("*")):
        relative = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            raise ProductError(f"predecessor namespace contains a link: {entry}")
        if entry.is_dir():
            entries[relative] = {
                "kind": "directory",
                "mode": entry.stat().st_mode & 0o777,
            }
        elif entry.is_file():
            entries[relative] = {
                "kind": "file",
                "mode": entry.stat().st_mode & 0o777,
                "size": entry.stat().st_size,
                "sha256": sha256_file(entry),
            }
        else:
            raise ProductError(
                f"predecessor namespace has unsupported entry: {entry}"
            )
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "entry_count": len(entries),
        "entries": entries,
    }


def validate_predecessor_namespaces() -> dict[str, Any]:
    manifest = load_object(PREDECESSOR_MANIFEST)
    baseline = manifest.get("before_construction")
    after = manifest.get("after_construction")
    current = {
        name: _namespace_manifest(path)
        for name, path in PREDECESSOR_NAMESPACES.items()
    }
    if (
        manifest.get("schema") != "ace2-r6-cf10-predecessor-byte-mode-manifest-v1"
        or manifest.get("status") != "PASS_CF07_CF08_CF09_BYTE_MODE_UNCHANGED"
        or baseline != after
        or current != baseline
    ):
        raise ProductError("CF07-CF09 predecessor byte/mode manifest differs")
    return {
        "status": manifest["status"],
        "manifest_sha256": sha256_file(PREDECESSOR_MANIFEST),
        "entry_counts": {
            name: record["entry_count"] for name, record in current.items()
        },
    }


'''
    text = replace_once(
        text,
        "def verify_qualification_bindings() -> dict[str, Any]:\n",
        metadata_helpers + "def verify_qualification_bindings() -> dict[str, Any]:\n",
        "tokenizer and predecessor authentication helpers",
    )
    text = text.replace(
        '"ace2-r6-cf09-endpoint-build-provenance-v1"',
        '"ace2-r6-cf10-endpoint-build-provenance-v1"',
    )
    return text


def make_driver() -> str:
    text = (SOURCE / "product_driver.py").read_text(encoding="utf-8")
    implementation = r'''def _fragment_metadata(token_form: str) -> dict[str, Any]:
    byte_fallback = re.fullmatch(r"(?:<0x[0-9A-Fa-f]{2}>)+", token_form)
    if byte_fallback:
        fragment = bytes.fromhex(
            "".join(re.findall(r"<0x([0-9A-Fa-f]{2})>", token_form))
        )
        source = "BYTE_FALLBACK_TOKEN"
    else:
        fragment = token_form.encode("utf-8", errors="strict")
        source = "TOKENIZER_DISPLAY_UTF8_NOT_FINAL_TEXT"
    try:
        fragment.decode("utf-8", errors="strict")
        status = "PARTIAL_FRAGMENT_READABLE_STRICT_UTF8_DIAGNOSTIC_ONLY"
    except UnicodeDecodeError:
        status = "PARTIAL_FRAGMENT_NOT_READABLE_STRICT_UTF8_DIAGNOSTIC_ONLY"
    return {
        "source": source,
        "escaped_bytes": "".join(f"\\x{value:02x}" for value in fragment),
        "byte_count": len(fragment),
        "partial_utf8_status": status,
        "used_as_final_text": False,
    }


def _validate_generated_sequence(
    token_ids: list[int], *, model_vocabulary_size: int, budget: int
) -> str:
    if not token_ids or len(token_ids) > budget:
        raise core.ProductError("generated token bookkeeping length differs")
    for ordinal, token_id in enumerate(token_ids):
        if (
            type(token_id) is not int
            or not 0 <= token_id < model_vocabulary_size
        ):
            raise core.ProductError(
                f"generated token ID is outside model vocabulary at step {ordinal}"
            )
    eos_positions = [
        ordinal
        for ordinal, token_id in enumerate(token_ids)
        if token_id in core.EOS_TOKEN_IDS
    ]
    if eos_positions:
        if eos_positions != [len(token_ids) - 1]:
            raise core.ProductError("EOS is not the final generated token")
        return "eos"
    if len(token_ids) != budget:
        raise core.ProductError("non-EOS generation does not contain four tokens")
    return "max_new_tokens"


def _validate_runtime_tokenizer(
    tokenizer: Any, domain: dict[str, Any]
) -> None:
    try:
        runtime_length = len(tokenizer)
        runtime_base_size = tokenizer.vocab_size
        runtime_mapping = tokenizer.get_vocab()
        runtime_added = tokenizer.get_added_vocab()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise core.ProductError(
            "official tokenizer runtime mapping is unavailable"
        ) from error
    if (
        runtime_length != domain["tokenizer_mapped_id_count"]
        or runtime_base_size != domain["tokenizer_base_vocabulary_size"]
        or runtime_mapping != domain["token_to_id"]
        or runtime_added != domain["added_token_to_id"]
    ):
        raise core.ProductError("official tokenizer runtime mapping differs")


def decode_token_ids_once(
    tokenizer: Any,
    token_ids: list[int],
    observability_path: Path,
    *,
    budget: int = core.GENERATION_BUDGET,
) -> tuple[str, str]:
    domain = core.authenticated_tokenizer_domain()
    evidence = core.tokenizer_domain_evidence()
    artifact = {
        "schema": "ace2-r6-cf10-predecode-observability-v1",
        "status": "PREDECODE_ATOMICALLY_PUBLISHED",
        "identity": core.ONE_USE_IDENTITY,
        "transaction_nonce": core.TRANSACTION_NONCE,
        "canonical_generation_result": {
            "kind": "COMPLETE_TOKEN_ID_SEQUENCE",
            "token_ids": list(token_ids),
            "token_count": len(token_ids),
            "generation_budget": budget,
        },
        "vocabulary_domains": evidence,
        "steps": [
            {
                "step_ordinal": ordinal,
                "token_id": token_id,
                "model_vocabulary_member": None,
                "tokenizer_mapped": None,
                "tokenizer_id_classification": "PENDING_MODEL_RANGE_VALIDATION",
                "cumulative_token_ids": list(token_ids[: ordinal + 1]),
            }
            for ordinal, token_id in enumerate(token_ids)
        ],
        "decode": {
            "api": "tokenizer.decode",
            "mode": {
                "whole_sequence": True,
                "call_count": 0,
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            },
            "classification": "PENDING_MODEL_RANGE_VALIDATION",
        },
        "protected_model_data": "NOT_RECORDED",
    }
    _atomic_json(observability_path, artifact, exclusive=False)
    try:
        stop_reason = _validate_generated_sequence(
            token_ids,
            model_vocabulary_size=core.MODEL_VOCABULARY_SIZE,
            budget=budget,
        )
    except core.ProductError as error:
        for step in artifact["steps"]:
            token_id = step["token_id"]
            in_range = (
                type(token_id) is int
                and 0 <= token_id < core.MODEL_VOCABULARY_SIZE
            )
            step["model_vocabulary_member"] = in_range
            step["tokenizer_id_classification"] = (
                "PENDING_TOKENIZER_MAPPING_CLASSIFICATION"
                if in_range
                else "MODEL_VOCABULARY_ID_OUT_OF_RANGE"
            )
        artifact["status"] = "FAIL_CLOSED"
        artifact["decode"]["classification"] = (
            "MODEL_VOCABULARY_ID_OUT_OF_RANGE"
            if any(
                step["tokenizer_id_classification"]
                == "MODEL_VOCABULARY_ID_OUT_OF_RANGE"
                for step in artifact["steps"]
            )
            else "INVALID_GENERATION_TOKEN_ID_SEQUENCE"
        )
        artifact["decode"]["validation_error"] = str(error)
        _atomic_json(observability_path, artifact, exclusive=False)
        raise
    artifact["canonical_generation_result"]["stop_reason"] = stop_reason
    unmapped: list[dict[str, int]] = []
    for step in artifact["steps"]:
        token_id = step["token_id"]
        mapped = token_id in domain["mapped_ids"]
        step["model_vocabulary_member"] = True
        step["tokenizer_mapped"] = mapped
        if token_id in domain["added_ids"]:
            classification = "MAPPED_ADDED_TOKEN_ID"
        elif mapped:
            classification = "MAPPED_BASE_TOKEN_ID"
        else:
            classification = "UNDECODABLE_RESERVED_MODEL_VOCAB_ID"
            unmapped.append(
                {
                    "step_ordinal": step["step_ordinal"],
                    "token_id": token_id,
                }
            )
        step["tokenizer_id_classification"] = classification
    artifact["decode"]["classification"] = (
        "UNDECODABLE_RESERVED_MODEL_VOCAB_ID"
        if unmapped
        else "PENDING_RUNTIME_TOKENIZER_AUTHENTICATION"
    )
    if unmapped:
        artifact["status"] = "FAIL_CLOSED"
        artifact["decode"]["undecodable_ids"] = unmapped
    _atomic_json(observability_path, artifact, exclusive=False)
    if unmapped:
        raise core.ProductError("UNDECODABLE_RESERVED_MODEL_VOCAB_ID")
    try:
        _validate_runtime_tokenizer(tokenizer, domain)
    except core.ProductError as error:
        artifact["status"] = "FAIL_CLOSED"
        artifact["decode"]["classification"] = (
            "TOKENIZER_MAPPING_AUTHENTICATION_FAILED"
        )
        artifact["decode"]["validation_error"] = str(error)
        _atomic_json(observability_path, artifact, exclusive=False)
        raise
    for step in artifact["steps"]:
        try:
            token_form = tokenizer.convert_ids_to_tokens(step["token_id"])
            if not isinstance(token_form, str):
                raise TypeError("token display is not text")
            fragment = _fragment_metadata(token_form)
        except (UnicodeEncodeError, TypeError, ValueError, RuntimeError) as error:
            artifact["status"] = "FAIL_CLOSED"
            artifact["decode"]["classification"] = (
                "TOKENIZER_DISPLAY_FORM_NOT_STRICT_UTF8"
            )
            artifact["decode"]["error_type"] = type(error).__name__
            _atomic_json(observability_path, artifact, exclusive=False)
            raise core.ProductError(
                "tokenizer display form is not strict UTF-8"
            ) from error
        step["tokenizer_token_display"] = token_form
        step["byte_fragment"] = fragment
    artifact["decode"]["classification"] = (
        "PENDING_OFFICIAL_WHOLE_SEQUENCE_DECODE"
    )
    _atomic_json(observability_path, artifact, exclusive=False)
    try:
        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except (UnicodeDecodeError, ValueError, TypeError, RuntimeError) as error:
        artifact["status"] = "FAIL_CLOSED"
        artifact["decode"]["mode"]["call_count"] = 1
        artifact["decode"]["classification"] = (
            "OFFICIAL_WHOLE_SEQUENCE_DECODE_EXCEPTION"
        )
        artifact["decode"]["error_type"] = type(error).__name__
        _atomic_json(observability_path, artifact, exclusive=False)
        raise core.ProductError(
            "official whole-sequence tokenizer decode failed"
        ) from error
    artifact["decode"]["mode"]["call_count"] = 1
    if not isinstance(decoded, str):
        classification = "OFFICIAL_DECODE_RETURNED_NON_TEXT"
    else:
        try:
            decoded.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            classification = "OFFICIAL_DECODE_NOT_STRICT_UTF8"
        else:
            if "\ufffd" in decoded:
                classification = "REPLACEMENT_CHARACTER_REJECTED"
            elif not core.readable_text(decoded):
                classification = "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8"
            else:
                classification = "PASS_READABLE_STRICT_UTF8"
    artifact["decode"]["classification"] = classification
    artifact["status"] = (
        "PASS_PREDECODE_OBSERVABILITY"
        if classification == "PASS_READABLE_STRICT_UTF8"
        else "FAIL_CLOSED"
    )
    _atomic_json(observability_path, artifact, exclusive=False)
    if classification != "PASS_READABLE_STRICT_UTF8":
        raise core.ProductError(
            f"whole-sequence decode rejected: {classification}"
        )
    return decoded, stop_reason


'''
    text = replace_region(
        text,
        "def _fragment_metadata(",
        "def run_host_oracles(",
        implementation,
        "tokenizer-domain decode implementation",
    )
    return text


def make_prepare() -> str:
    text = (SOURCE / "prepare_package.py").read_text(encoding="utf-8")
    text = replace_region(
        text,
        "def authorized_package_checks(",
        "def write_authorization(",
        "",
        "obsolete execution-authority preparation helpers",
    )
    text = text.replace(
        '"ace2-r6-cf09-endpoint-build-provenance-v1"',
        '"ace2-r6-cf10-endpoint-build-provenance-v1"',
    )
    text = text.replace(
        '"CF09_LOCAL_TEMPORARY_REMOVED"', '"CF10_LOCAL_TEMPORARY_REMOVED"'
    )
    text = text.replace('prefix=".cf09-repro-"', 'prefix=".cf10-repro-"')
    text = replace_once(
        text,
        '            "predecode_observability_atomic": True,\n',
        '''            "predecode_observability_atomic": True,
            "model_vocabulary_dimension_authenticated": True,
            "tokenizer_base_vocabulary_authenticated": True,
            "tokenizer_mapped_decode_domain_authenticated": True,
            "reserved_model_vocab_ids_fail_closed": True,
            "observability_precedes_decodability_rejection": True,
            "cf07_cf08_cf09_byte_mode_preserved": True,
''',
        "CF10 package checks",
    )
    text = replace_once(
        text,
        '            "tokenizer_config_sha256": core.EXPECTED["tokenizer_config"],\n',
        '''            "tokenizer_config_sha256": core.EXPECTED["tokenizer_config"],
            "tokenizer_domain": gates["tokenizer_domain"],
''',
        "tokenizer-domain authorization binding",
    )
    return text


TESTS = r'''from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import package_core as core
import product_driver as driver


class FakeTokenizer:
    def __init__(
        self,
        decoded: object = "test",
        error: Exception | None = None,
        *,
        mutate_mapping: bool = False,
    ) -> None:
        self.decoded = decoded
        self.error = error
        self.decode_calls: list[list[int]] = []
        self.domain = core.authenticated_tokenizer_domain()
        self.vocab_size = core.TOKENIZER_BASE_VOCABULARY_SIZE
        self.mutate_mapping = mutate_mapping
        self.forms = {
            1: "A",
            2: "<0xE2>",
            3: "<0x82>",
            4: "<0xAC>",
            5: "world",
            151643: "<|endoftext|>",
            151645: "<|im_end|>",
        }

    def __len__(self) -> int:
        return core.TOKENIZER_MAPPED_ID_COUNT

    def get_vocab(self) -> dict[str, int]:
        mapping = self.domain["token_to_id"]
        if not self.mutate_mapping:
            return mapping
        changed = dict(mapping)
        first = next(iter(changed))
        changed[first] = core.TOKENIZER_MAPPED_ID_COUNT
        return changed

    def get_added_vocab(self) -> dict[str, int]:
        return self.domain["added_token_to_id"]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.forms.get(token_id, f"token-{token_id}")

    def decode(self, token_ids: list[int], **kwargs: object) -> object:
        self.decode_calls.append(list(token_ids))
        if self.error is not None:
            raise self.error
        return self.decoded


class ProductPackageTests(unittest.TestCase):
    def decode(
        self, tokenizer: FakeTokenizer, token_ids: list[int], *, budget: int = 4
    ) -> tuple[str, dict[str, object]]:
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "predecode.json"
            text, _ = driver.decode_token_ids_once(
                tokenizer, token_ids, path, budget=budget
            )
            return text, json.loads(path.read_text(encoding="utf-8"))

    def test_real_authenticated_tokenizer_metadata_has_three_domains(self) -> None:
        domain = core.authenticated_tokenizer_domain()
        self.assertEqual(domain["model_vocabulary_size"], 151936)
        self.assertEqual(domain["tokenizer_base_vocabulary_size"], 151643)
        self.assertEqual(domain["tokenizer_mapped_id_count"], 151665)
        self.assertEqual(domain["tokenizer_added_token_count"], 22)
        self.assertEqual(domain["tokenizer_maximum_mapped_id"], 151664)
        self.assertEqual(len(domain["mapped_ids"]), 151665)
        self.assertEqual(max(domain["mapped_ids"]), 151664)
        self.assertNotEqual(len(FakeTokenizer()), core.MODEL_VOCABULARY_SIZE)

    def test_mapped_base_id_uses_one_whole_sequence_decode(self) -> None:
        tokenizer = FakeTokenizer("ASCII")
        decoded, artifact = self.decode(tokenizer, [1, 1, 1, 1])
        self.assertEqual(decoded, "ASCII")
        self.assertEqual(tokenizer.decode_calls, [[1, 1, 1, 1]])
        self.assertEqual(
            artifact["decode"]["classification"], "PASS_READABLE_STRICT_UTF8"
        )
        self.assertTrue(
            all(
                step["tokenizer_id_classification"] == "MAPPED_BASE_TOKEN_ID"
                for step in artifact["steps"]
            )
        )

    def test_mapped_added_special_token_is_decodable(self) -> None:
        tokenizer = FakeTokenizer("done")
        _, artifact = self.decode(tokenizer, [5, 151645])
        self.assertEqual(tokenizer.decode_calls, [[5, 151645]])
        self.assertEqual(
            artifact["steps"][-1]["tokenizer_id_classification"],
            "MAPPED_ADDED_TOKEN_ID",
        )
        self.assertEqual(
            artifact["canonical_generation_result"]["stop_reason"], "eos"
        )

    def test_gap_and_reserved_model_vocab_ids_fail_without_decode(self) -> None:
        for token_id in (151665, 151700, 151935):
            with self.subTest(token_id=token_id):
                tokenizer = FakeTokenizer()
                with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
                    path = Path(temporary) / "predecode.json"
                    with self.assertRaisesRegex(
                        core.ProductError,
                        "UNDECODABLE_RESERVED_MODEL_VOCAB_ID",
                    ):
                        driver.decode_token_ids_once(
                            tokenizer, [1, 1, 1, token_id], path
                        )
                    artifact = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(tokenizer.decode_calls, [])
                    self.assertEqual(
                        artifact["decode"]["classification"],
                        "UNDECODABLE_RESERVED_MODEL_VOCAB_ID",
                    )
                    self.assertEqual(
                        artifact["canonical_generation_result"]["token_ids"],
                        [1, 1, 1, token_id],
                    )
                    self.assertEqual(
                        artifact["decode"]["undecodable_ids"][0]["token_id"],
                        token_id,
                    )

    def test_model_out_of_range_ids_are_classified_before_decode(self) -> None:
        for token_id in (-1, core.MODEL_VOCABULARY_SIZE):
            with self.subTest(token_id=token_id):
                tokenizer = FakeTokenizer()
                with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
                    path = Path(temporary) / "predecode.json"
                    with self.assertRaisesRegex(
                        core.ProductError, "outside model vocabulary"
                    ):
                        driver.decode_token_ids_once(
                            tokenizer, [1, 1, 1, token_id], path
                        )
                    artifact = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(tokenizer.decode_calls, [])
                    self.assertEqual(
                        artifact["decode"]["classification"],
                        "MODEL_VOCABULARY_ID_OUT_OF_RANGE",
                    )

    def test_multibyte_unicode_split_across_byte_fallback_pieces(self) -> None:
        tokenizer = FakeTokenizer("€ value")
        decoded, artifact = self.decode(tokenizer, [2, 3, 4, 151645])
        self.assertEqual(decoded, "€ value")
        self.assertEqual(tokenizer.decode_calls, [[2, 3, 4, 151645]])
        fragments = [step["byte_fragment"] for step in artifact["steps"][:3]]
        self.assertTrue(
            all(item["source"] == "BYTE_FALLBACK_TOKEN" for item in fragments)
        )
        self.assertTrue(
            all(item["used_as_final_text"] is False for item in fragments)
        )
        self.assertEqual(
            fragments[0]["partial_utf8_status"],
            "PARTIAL_FRAGMENT_NOT_READABLE_STRICT_UTF8_DIAGNOSTIC_ONLY",
        )

    def test_decode_exception_retains_complete_observability(self) -> None:
        tokenizer = FakeTokenizer(
            error=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        )
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "predecode.json"
            with self.assertRaisesRegex(
                core.ProductError, "official whole-sequence"
            ):
                driver.decode_token_ids_once(tokenizer, [1, 1, 1, 1], path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["decode"]["classification"],
                "OFFICIAL_WHOLE_SEQUENCE_DECODE_EXCEPTION",
            )
            self.assertEqual(artifact["decode"]["mode"]["call_count"], 1)
            self.assertEqual(
                artifact["canonical_generation_result"]["token_ids"],
                [1, 1, 1, 1],
            )
            self.assertFalse((Path(temporary) / "product-result.json").exists())

    def test_lone_surrogate_is_rejected_as_non_strict_utf8(self) -> None:
        tokenizer = FakeTokenizer("\ud800")
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "predecode.json"
            with self.assertRaisesRegex(
                core.ProductError, "OFFICIAL_DECODE_NOT_STRICT_UTF8"
            ):
                driver.decode_token_ids_once(tokenizer, [1, 1, 1, 1], path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["decode"]["classification"],
                "OFFICIAL_DECODE_NOT_STRICT_UTF8",
            )

    def test_replacement_character_is_rejected(self) -> None:
        tokenizer = FakeTokenizer("\ufffd")
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "predecode.json"
            with self.assertRaisesRegex(
                core.ProductError, "REPLACEMENT_CHARACTER"
            ):
                driver.decode_token_ids_once(tokenizer, [1, 1, 1, 1], path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["decode"]["classification"],
                "REPLACEMENT_CHARACTER_REJECTED",
            )

    def test_runtime_tokenizer_mapping_tamper_is_rejected(self) -> None:
        tokenizer = FakeTokenizer(mutate_mapping=True)
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "predecode.json"
            with self.assertRaisesRegex(
                core.ProductError, "runtime mapping differs"
            ):
                driver.decode_token_ids_once(tokenizer, [1, 1, 1, 1], path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(tokenizer.decode_calls, [])
            self.assertEqual(
                artifact["decode"]["classification"],
                "TOKENIZER_MAPPING_AUTHENTICATION_FAILED",
            )

    def test_eos_and_four_token_generation_bookkeeping(self) -> None:
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "short.json"
            with self.assertRaisesRegex(
                core.ProductError, "does not contain four"
            ):
                driver.decode_token_ids_once(FakeTokenizer(), [1, 1, 1], path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["canonical_generation_result"]["token_ids"], [1, 1, 1]
            )

    def test_arbitrary_utf8_input_and_readability(self) -> None:
        value = "Grüße 世界 — registers"
        self.assertEqual(
            value.encode("utf-8").decode("utf-8", errors="strict"), value
        )
        self.assertTrue(core.readable_text(value))
        self.assertFalse(core.readable_text("\ufffd"))

    def test_atomic_result_publication_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "product-result.json"
            driver.write_json_atomic_exclusive(path, {"status": "PASS"})
            self.assertEqual(json.loads(path.read_text())["status"], "PASS")
            with self.assertRaises(FileExistsError):
                driver.write_json_atomic_exclusive(path, {"status": "REPLACED"})
            self.assertEqual(json.loads(path.read_text())["status"], "PASS")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*")), [])

    def test_source_model_tokenizer_export_and_rtl_tamper_rejection(self) -> None:
        bindings = {
            "checkpoint_sha256": core.EXPECTED["checkpoint"],
            "export_manifest_sha256": core.EXPECTED["export_manifest"],
            "packed_w4_sha256": core.EXPECTED["packed_w4"],
            "weight_scale32_sha256": core.EXPECTED["weight_scale32"],
            "ordered_rows_sha256": core.EXPECTED["ordered_rows"],
            "per_row_digest_ledger_sha256": core.EXPECTED["row_digests"],
            "tokenizer_json_sha256": core.EXPECTED["tokenizer_json"],
            "tokenizer_config_sha256": core.EXPECTED["tokenizer_config"],
            "tokenizer_domain": core.tokenizer_domain_evidence(),
            "official_model_sha256": core.EXPECTED["model"],
            "rtl_source_closure": core.rtl_source_closure(),
            "public_ace2_shell_interface": core.public_ace2_shell_interface(),
        }
        cases = {
            "model": "official_model_sha256",
            "tokenizer": "tokenizer_json_sha256",
            "tokenizer-domain": "tokenizer_domain",
            "export": "export_manifest_sha256",
            "rtl": "rtl_source_closure",
        }
        for label, field in cases.items():
            with self.subTest(label=label):
                mutated = copy.deepcopy(bindings)
                mutated[field] = {} if field in {"rtl_source_closure", "tokenizer_domain"} else "0" * 64
                with self.assertRaisesRegex(core.ProductError, field):
                    core.validate_binding_snapshot(mutated)
        source_bindings = {
            (core.PACKAGE / name).relative_to(core.ROOT).as_posix():
            core.sha256_file(core.PACKAGE / name)
            for name in core.PACKAGE_SOURCE_NAMES
        }
        first = sorted(source_bindings)[0]
        source_bindings[first] = "0" * 64
        with self.assertRaisesRegex(core.ProductError, "hash-mismatched"):
            core.validate_package_source_bindings(
                {"package_sources": source_bindings}
            )

    def test_cf09_terminal_zero_authority_and_predecessors_are_authenticated(
        self,
    ) -> None:
        terminal = core.validate_cf09_terminal_failure()
        self.assertEqual(
            terminal["status"],
            "AUTHENTICATED_TOKENIZER_DOMAIN_CONTRACT_DEFECT",
        )
        self.assertTrue(terminal["authority_consumed"])
        self.assertEqual(
            core.validate_construction_only_record()["authority_cardinality"], 0
        )
        predecessors = core.validate_predecessor_namespaces()
        self.assertEqual(
            predecessors["status"],
            "PASS_CF07_CF08_CF09_BYTE_MODE_UNCHANGED",
        )
        with self.assertRaisesRegex(
            core.ProductError, "no execution authority"
        ):
            core.validate_authorization()

    def test_live_rtl_interface_provenance_is_exact(self) -> None:
        closure = core.rtl_source_closure()
        interface = core.public_ace2_shell_interface()
        self.assertEqual(len(closure["files"]), 19)
        self.assertEqual(interface["parameter_count"], 14)
        self.assertEqual(interface["port_count"], 64)

    def test_prepared_package_two_build_reproducibility_and_absence(self) -> None:
        if not core.AUTHORIZATION.exists():
            self.skipTest("package is not materialized yet")
        authorization = core.validate_prepared_authorization()
        self.assertEqual(authorization["authorization_state"], "NOT_GRANTED")
        self.assertEqual(authorization["execution_limit"], 0)
        report = core.load_object(core.REPRODUCIBILITY_REPORT)
        self.assertEqual(report["status"], "PASS_TWO_BUILD_BYTE_MODE_EQUAL")
        self.assertEqual(report["build_count"], 2)
        core.assert_fresh_execution_state()
        for path in (
            core.ATTEMPT,
            core.CONSUMED,
            core.PACKAGE / "execution-registry.json",
            core.PACKAGE / "endpoint-invocation.json",
            core.PACKAGE / "completion",
            core.PACKAGE / "completion-summary.json",
            core.PACKAGE / "product-result.json",
        ):
            self.assertFalse(path.exists(), str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


def main() -> int:
    if PACKAGE.exists():
        raise RuntimeError(f"fresh CF10 namespace already exists: {PACKAGE}")
    required = (
        SOURCE / "package_core.py",
        SOURCE / "prepare_package.py",
        SOURCE / "product_driver.py",
        SOURCE / "launch.sh",
        SOURCE / "input.txt",
        TERMINAL_HANDOFF / "latest.json",
        TERMINAL_HANDOFF / "round-0001.json",
        TERMINAL_SEAL,
        *PREDECESSORS.values(),
    )
    for path in required:
        if not path.exists() or path.is_symlink():
            raise RuntimeError(f"immutable prerequisite is absent or linked: {path}")
    before = {
        name: namespace_manifest(path) for name, path in PREDECESSORS.items()
    }
    PACKAGE.mkdir(parents=True)
    (PACKAGE / "authority").mkdir()
    (PACKAGE / "tests").mkdir()
    (PACKAGE / "package_core.py").write_text(make_core(), encoding="utf-8")
    (PACKAGE / "prepare_package.py").write_text(make_prepare(), encoding="utf-8")
    (PACKAGE / "product_driver.py").write_text(make_driver(), encoding="utf-8")
    (PACKAGE / "tests/test_product_package.py").write_text(
        TESTS, encoding="utf-8"
    )
    shutil.copyfile(SOURCE / "input.txt", PACKAGE / "input.txt")
    shutil.copyfile(SOURCE / "launch.sh", PACKAGE / "launch.sh")
    (PACKAGE / "authority/manager-decision.json").write_bytes(
        canonical_bytes(construction_record())
    )
    after = {
        name: namespace_manifest(path) for name, path in PREDECESSORS.items()
    }
    if after != before:
        raise RuntimeError("CF07-CF09 predecessor namespace changed during construction")
    predecessor_record = {
        "schema": "ace2-r6-cf10-predecessor-byte-mode-manifest-v1",
        "status": "PASS_CF07_CF08_CF09_BYTE_MODE_UNCHANGED",
        "before_construction": before,
        "after_construction": after,
    }
    (PACKAGE / "predecessor-manifest.json").write_bytes(
        canonical_bytes(predecessor_record)
    )
    modes = {
        "package_core.py": 0o444,
        "prepare_package.py": 0o555,
        "product_driver.py": 0o444,
        "tests/test_product_package.py": 0o444,
        "input.txt": 0o444,
        "launch.sh": 0o555,
        "authority/manager-decision.json": 0o444,
        "predecessor-manifest.json": 0o444,
    }
    for name, mode in modes.items():
        (PACKAGE / name).chmod(mode)
    print(PACKAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
