#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf08"
PACKAGE = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf09"
TERMINAL_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6stage1terminalseal08"
)
TERMINAL_SEAL = SOURCE / "terminal-seal.json"
IDENTITY = "stage1-w4a8-r6-product-prep-cf09-attempt-0001"
NONCE = "1f4f1754cc8bce90e836c29bdd57e03b9e5e927244d76cad97bf039e92abcb7f"
CONSTRUCTION_DECISION = "mgr-r6-cf09-decode-observability-package-01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def make_core() -> str:
    text = (SOURCE / "package_core.py").read_text(encoding="utf-8")
    text = text.replace(
        '"stage1-w4a8-r6-product-prep-cf08-attempt-0001"', f'"{IDENTITY}"'
    )
    text = text.replace(
        '"d4a3b681807f1d2f7aca74ba9c868cbcfbfad141b03ca877e624b994361bcf79"',
        f'"{NONCE}"',
    )
    text = text.replace(
        '"r6-final-export-product-cf08"', '"r6-final-export-product-cf09"'
    )
    text = text.replace(
        '    "authority/manager-execution-authority.json",\n', ""
    )
    text = text.replace(
        'AUTHORITY_RECORD = PACKAGE / "authority/manager-execution-authority.json"',
        'AUTHORITY_RECORD = PACKAGE / "authority/manager-execution-authority.json"',
    )
    insert_after = 'EXECUTION_REVIEW_MISSION_ID = "r6executionauthorityfix08"\n'
    terminal_constants = f'''
CF08_TERMINAL_LATEST = Path({str(TERMINAL_HANDOFF / "latest.json")!r})
CF08_TERMINAL_SEAL = Path({str(TERMINAL_SEAL)!r})
CF08_TERMINAL_LATEST_SHA256 = {sha256(TERMINAL_HANDOFF / "latest.json")!r}
CF08_TERMINAL_HANDOFF_SHA256 = {sha256(TERMINAL_HANDOFF / "round-0002.json")!r}
CF08_TERMINAL_SEAL_SHA256 = {sha256(TERMINAL_SEAL)!r}
CF08_TERMINAL_MISSION_ID = "r6stage1terminalseal08"
VOCABULARY_SIZE = 151936
PREDECODE_OBSERVABILITY = PACKAGE / "attempt-0001/predecode-observability.json"
'''
    text = replace_once(
        text,
        insert_after,
        insert_after + terminal_constants,
        "terminal constants",
    )
    authority_region = '''def construction_only_decision_record() -> dict[str, Any]:
    return {
        "schema": "ace2-r6-external-manager-authority-v1",
        "status": "GRANTED",
        "authority_cardinality": 1,
        "semantic_fields": {
            "decision_id": "mgr-r6-cf09-decode-observability-package-01",
            "scope": "r6decodeobservabilityprep09",
            "authorized_action": "CONSTRUCTION_AND_INDEPENDENT_REVIEW_ONLY",
            "sealed_cf07_identity": (
                "stage1-w4a8-r6-product-prep-cf07-attempt-0001"
            ),
            "sealed_cf08_identity": (
                "stage1-w4a8-r6-product-prep-cf08-attempt-0001"
            ),
            "sealed_cf08_outcome": "SEALED_RUNTIME_FAILURE",
            "sealed_cf08_retry": "FORBIDDEN",
            "sealed_cf08_replay": "FORBIDDEN",
            "sealed_cf08_resume": "FORBIDDEN",
            "sealed_cf08_relaunch": "FORBIDDEN",
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
            "independent_l2_review": "REQUIRED",
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
    raise ProductError("CF09 has no execution authority")


def validate_manager_authority() -> dict[str, Any]:
    raise ProductError("CF09 has no execution authority")


def _validate_immutable_authority_record(
    path: Path, expected: dict[str, Any], label: str
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProductError(f"{label} is absent or linked")
    record = load_object(path)
    if canonical_bytes(record) != canonical_bytes(expected):
        raise ProductError(f"{label} semantic fields differ")
    if path.stat().st_mode & 0o222:
        raise ProductError(f"{label} is mutable")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "semantic_fields": expected["semantic_fields"],
    }


def validate_cf08_terminal_failure() -> dict[str, Any]:
    require_file(
        CF08_TERMINAL_LATEST,
        CF08_TERMINAL_LATEST_SHA256,
        "CF08 terminal latest handoff",
    )
    review = validate_reviewer_gate(
        CF08_TERMINAL_LATEST, CF08_TERMINAL_MISSION_ID
    )
    if review["handoff_sha256"] != CF08_TERMINAL_HANDOFF_SHA256:
        raise ProductError("CF08 terminal reviewed handoff differs")
    require_file(
        CF08_TERMINAL_SEAL,
        CF08_TERMINAL_SEAL_SHA256,
        "CF08 terminal seal",
    )
    seal = load_object(CF08_TERMINAL_SEAL)
    identity = seal.get("identity", {})
    terminal = seal.get("terminal", {})
    authorization = seal.get("authorization", {})
    artifacts = seal.get("execution_artifacts", {})
    if (
        seal.get("status") != "SEALED_RUNTIME_FAILURE"
        or seal.get("first_natural_terminal") is not True
        or seal.get("durable_process_starts") != 1
        or identity.get("package_attempt")
        != "stage1-w4a8-r6-product-prep-cf08-attempt-0001"
        or terminal.get("exit_code") != 2
        or terminal.get("stderr")
        != "product execution refused: whole-sequence decode is not readable strict UTF-8\\n"
        or terminal.get("stdout") != ""
        or terminal.get("timeout_triggered") is not False
        or terminal.get("orphaned") is not False
        or seal.get("process", {}).get("durable_elapsed_seconds") != 1363.5
        or authorization.get("consumption_status")
        != "CONSUMED_BEFORE_PRODUCT_EXECUTION"
        or authorization.get("retry") != "FORBIDDEN"
        or authorization.get("replay") != "FORBIDDEN"
        or authorization.get("resume") != "FORBIDDEN"
        or authorization.get("relaunch") != "FORBIDDEN"
        or artifacts.get("attempt_0001", {}).get("status") != "ABSENT"
        or artifacts.get("completion_summary", {}).get("status") != "ABSENT"
        or artifacts.get("product_result", {}).get("status") != "ABSENT"
    ):
        raise ProductError("CF08 terminal failure semantics differ")
    return {
        **review,
        "terminal_seal": str(CF08_TERMINAL_SEAL),
        "terminal_seal_sha256": CF08_TERMINAL_SEAL_SHA256,
        "status": "AUTHENTICATED_SEALED_RUNTIME_FAILURE",
        "exit_code": 2,
        "elapsed_seconds": 1363.5,
        "decode_failure": "whole-sequence decode is not readable strict UTF-8",
        "authority_consumed": True,
        "product_result": "ABSENT",
        "token_result": "ABSENT",
    }


'''
    text = replace_region(
        text,
        "def construction_only_decision_record()",
        "def validate_execution_argv(",
        authority_region,
        "authority implementation",
    )
    old_gates = '''        "qualification_binding": verify_qualification_bindings(),
        "manager_execution_authority": validate_manager_authority(),
        "construction_only_record": validate_construction_only_record(),
'''
    new_gates = '''        "qualification_binding": verify_qualification_bindings(),
        "cf08_terminal_failure": validate_cf08_terminal_failure(),
        "construction_only_record": validate_construction_only_record(),
'''
    text = replace_once(text, old_gates, new_gates, "prerequisite gates")
    claims_region = '''def commit_authorization_claim(path: Path = CONSUMED) -> None:
    raise ProductError("CF09 has no authority to consume")


def claim_authorization(
    authorization: dict[str, Any], path: Path = CONSUMED
) -> None:
    raise ProductError("CF09 has no authority to consume")


'''
    text = replace_region(
        text,
        "def commit_authorization_claim(",
        "AUTHORIZATION_KEYS =",
        claims_region,
        "consumption functions",
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
        "authorization validation",
    )
    text = text.replace(
        'raise ProductError(f"CF08 execution state is not fresh: {present}")',
        'raise ProductError(f"CF09 execution state is not fresh: {present}")',
    )
    old_literals = '''    expected_bindings = {
        "checkpoint_sha256": EXPECTED["checkpoint"],
        "export_manifest_sha256": EXPECTED["export_manifest"],
        "packed_w4_sha256": EXPECTED["packed_w4"],
        "weight_scale32_sha256": EXPECTED["weight_scale32"],
        "ordered_rows_sha256": EXPECTED["ordered_rows"],
        "per_row_digest_ledger_sha256": EXPECTED["row_digests"],
        "tokenizer_json_sha256": EXPECTED["tokenizer_json"],
        "tokenizer_config_sha256": EXPECTED["tokenizer_config"],
    }
    for field, expected in expected_bindings.items():
        if bindings.get(field) != expected:
            raise ProductError(f"authorization literal binding differs: {field}")
'''
    text = replace_once(
        text, old_literals, "    validate_binding_snapshot(bindings)\n", "binding snapshot"
    )
    text = replace_region(
        text,
        "def validate_authorization()",
        "def host_time()",
        '''def validate_authorization() -> dict[str, Any]:
    raise ProductError("CF09 has no execution authority")


def validate_execution_review() -> dict[str, Any]:
    raise ProductError("CF09 execution authority is a separate later mission")


''',
        "execution authorization gate",
    )
    text = text.replace(
        '"ace2-r6-cf08-endpoint-build-provenance-v1"',
        '"ace2-r6-cf09-endpoint-build-provenance-v1"',
    )
    return text


def make_driver() -> str:
    text = (SOURCE / "product_driver.py").read_text(encoding="utf-8")
    text = text.replace("import json\n", "import json\nimport os\nimport tempfile\n")
    implementation = r'''def _atomic_json(path: Path, value: Any, *, exclusive: bool) -> None:
    raw = core.canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic_exclusive(path: Path, value: Any) -> None:
    _atomic_json(path, value, exclusive=True)


def _fragment_metadata(token_form: str) -> dict[str, Any]:
    byte_fallback = re.fullmatch(r"(?:<0x[0-9A-Fa-f]{2}>)+", token_form)
    if byte_fallback:
        fragment = bytes.fromhex("".join(re.findall(r"<0x([0-9A-Fa-f]{2})>", token_form)))
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
    token_ids: list[int], *, vocabulary_size: int, budget: int
) -> str:
    if not token_ids or len(token_ids) > budget:
        raise core.ProductError("generated token bookkeeping length differs")
    for ordinal, token_id in enumerate(token_ids):
        if type(token_id) is not int or not 0 <= token_id < vocabulary_size:
            raise core.ProductError(
                f"generated token ID is outside vocabulary at step {ordinal}"
            )
    eos_positions = [
        ordinal for ordinal, token_id in enumerate(token_ids)
        if token_id in core.EOS_TOKEN_IDS
    ]
    if eos_positions:
        if eos_positions != [len(token_ids) - 1]:
            raise core.ProductError("EOS is not the final generated token")
        return "eos"
    if len(token_ids) != budget:
        raise core.ProductError("non-EOS generation does not contain four tokens")
    return "max_new_tokens"


def decode_token_ids_once(
    tokenizer: Any,
    token_ids: list[int],
    observability_path: Path,
    *,
    budget: int = core.GENERATION_BUDGET,
) -> tuple[str, str]:
    vocabulary_size = len(tokenizer)
    if vocabulary_size != core.VOCABULARY_SIZE:
        raise core.ProductError("official tokenizer vocabulary bounds differ")
    validation_artifact = {
        "schema": "ace2-r6-cf09-predecode-observability-v1",
        "status": "PREDECODE_ATOMICALLY_PUBLISHED",
        "identity": core.ONE_USE_IDENTITY,
        "transaction_nonce": core.TRANSACTION_NONCE,
        "canonical_generation_result": {
            "kind": "COMPLETE_TOKEN_ID_SEQUENCE",
            "token_ids": list(token_ids),
            "token_count": len(token_ids),
            "generation_budget": budget,
        },
        "steps": [],
        "decode": {
            "api": "tokenizer.decode",
            "mode": {
                "whole_sequence": True,
                "call_count": 0,
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            },
            "classification": "PENDING_TOKEN_ID_VALIDATION",
        },
        "protected_model_data": "NOT_RECORDED",
    }
    _atomic_json(observability_path, validation_artifact, exclusive=False)
    try:
        stop_reason = _validate_generated_sequence(
            token_ids, vocabulary_size=vocabulary_size, budget=budget
        )
    except core.ProductError as error:
        validation_artifact["status"] = "FAIL_CLOSED"
        validation_artifact["decode"]["classification"] = (
            "INVALID_GENERATION_TOKEN_ID_SEQUENCE"
        )
        validation_artifact["decode"]["validation_error"] = str(error)
        _atomic_json(observability_path, validation_artifact, exclusive=False)
        raise
    steps = []
    cumulative: list[int] = []
    for ordinal, token_id in enumerate(token_ids):
        cumulative.append(token_id)
        token_form = tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(token_form, str):
            raise core.ProductError(
                f"tokenizer display form is invalid at step {ordinal}"
            )
        steps.append(
            {
                "step_ordinal": ordinal,
                "token_id": token_id,
                "vocabulary_bounds": {"minimum": 0, "exclusive_maximum": vocabulary_size},
                "tokenizer_token_display": token_form,
                "byte_fragment": _fragment_metadata(token_form),
                "cumulative_token_ids": list(cumulative),
            }
        )
    artifact = {
        "schema": "ace2-r6-cf09-predecode-observability-v1",
        "status": "PREDECODE_ATOMICALLY_PUBLISHED",
        "identity": core.ONE_USE_IDENTITY,
        "transaction_nonce": core.TRANSACTION_NONCE,
        "canonical_generation_result": {
            "kind": "COMPLETE_TOKEN_ID_SEQUENCE",
            "token_ids": list(token_ids),
            "token_count": len(token_ids),
            "generation_budget": budget,
            "stop_reason": stop_reason,
        },
        "steps": steps,
        "decode": {
            "api": "tokenizer.decode",
            "mode": {
                "whole_sequence": True,
                "call_count": 0,
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            },
            "classification": "PENDING_OFFICIAL_WHOLE_SEQUENCE_DECODE",
        },
        "protected_model_data": "NOT_RECORDED",
    }
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
        artifact["decode"]["classification"] = "OFFICIAL_WHOLE_SEQUENCE_DECODE_EXCEPTION"
        artifact["decode"]["error_type"] = type(error).__name__
        _atomic_json(observability_path, artifact, exclusive=False)
        raise core.ProductError("official whole-sequence tokenizer decode failed") from error
    artifact["decode"]["mode"]["call_count"] = 1
    if not isinstance(decoded, str):
        classification = "OFFICIAL_DECODE_RETURNED_NON_TEXT"
    elif "\ufffd" in decoded:
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


def run_host_oracles(
    runtime: Any, prompt: str, expected_input_ids: list[int], budget: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, dict[str, float]]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    cpu_start, wall_start = core.host_time()
    cached = []
    generated: list[int] = []
    boundary = runtime.prefill(messages, expected_input_ids=expected_input_ids)
    for step in range(budget):
        logits = boundary["logits"]
        token = max(range(len(logits)), key=logits.__getitem__)
        cached.append(
            {
                "step": step,
                "absolute_position": len(expected_input_ids) - 1 + step,
                "token_id": token,
                "logits_sha256": core.logits_sha256(logits),
                "layer_state_digests": boundary["layer_state_digests"],
            }
        )
        generated.append(token)
        if token in core.EOS_TOKEN_IDS:
            break
        if step + 1 < budget:
            boundary = runtime.decode(token)

    device = next(runtime.model.parameters()).device
    free = []
    sequence = list(expected_input_ids)
    for step, cached_step in enumerate(cached):
        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = runtime.model(input_ids=input_ids, use_cache=False)
        logits = output.logits[:, -1].detach().cpu().reshape(-1).tolist()
        token = max(range(len(logits)), key=logits.__getitem__)
        free.append(
            {
                "step": step,
                "absolute_position": len(expected_input_ids) - 1 + step,
                "token_id": token,
                "logits_sha256": core.logits_sha256(logits),
            }
        )
        sequence.append(token)
    core.compare_host_oracles(free, cached)
    decoded, stop_reason = decode_token_ids_once(
        runtime.tokenizer,
        generated,
        core.PREDECODE_OBSERVABILITY,
        budget=budget,
    )
    timing = {
        "host_cpu_seconds": time.process_time() - cpu_start,
        "host_wall_seconds": time.perf_counter() - wall_start,
    }
    return free, cached, decoded, stop_reason, timing


'''
    implementation = implementation.replace("import re", "import re")
    text = replace_region(
        text,
        "def run_host_oracles(",
        "def load_jsonl(",
        implementation,
        "decode implementation",
    )
    text = text.replace("import tempfile\n", "import tempfile\nimport re\n", 1)
    text = replace_once(
        text,
        "    core.write_json_exclusive(core.PACKAGE / \"product-result.json\", result)\n",
        "    write_json_atomic_exclusive(core.PACKAGE / \"product-result.json\", result)\n",
        "atomic product result",
    )
    return text


def make_prepare() -> str:
    text = (SOURCE / "prepare_package.py").read_text(encoding="utf-8")
    text = text.replace(
        '"ace2-r6-cf08-endpoint-build-provenance-v1"',
        '"ace2-r6-cf09-endpoint-build-provenance-v1"',
    )
    text = text.replace(
        '"CF08_LOCAL_TEMPORARY_REMOVED"', '"CF09_LOCAL_TEMPORARY_REMOVED"'
    )
    text = text.replace('prefix=".cf08-repro-"', 'prefix=".cf09-repro-"')
    text = text.replace(
        '"external_authorization_granted": True,',
        '"external_authorization_granted": False,',
    )
    text = text.replace('"authority_cardinality": 1,', '"authority_cardinality": 0,')
    text = text.replace(
        '"execution_review_completed": False,',
        '"execution_review_completed": False,\n'
        '            "token_ids_are_canonical_generation_result": True,\n'
        '            "official_whole_sequence_decode_exactly_once": True,\n'
        '            "predecode_observability_atomic": True,',
    )
    text = text.replace(
        '''            "manager_execution_authority_sha256": core.sha256_file(
                core.AUTHORITY_RECORD
            ),
''',
        '''            "manager_construction_authority_sha256": core.sha256_file(
                core.CONSTRUCTION_AUTHORITY_RECORD
            ),
''',
    )
    writer = '''def write_authorization(output: Path, gates: dict[str, object]) -> None:
    construction = gates["construction_only_record"]
    source_hashes = {
        (core.PACKAGE / name).relative_to(core.ROOT).as_posix(): core.sha256_file(
            core.PACKAGE / name
        )
        for name in core.PACKAGE_SOURCE_NAMES
    }
    tokenizer_config = core.load_object(core.TOKENIZER_ROOT / "tokenizer_config.json")
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise core.ProductError("official chat template is absent")
    qualification = gates["qualification_binding"]
    authorization = {
        "schema": "ace2-r6-stage1-product-authorization-sidecar-v1",
        "status": "PREPARED_NOT_AUTHORIZED_NOT_CONSUMED",
        "authorization_state": "NOT_GRANTED",
        "authority_provenance": {
            "origin": "CONSTRUCTION_ONLY_MANAGER_DECISION",
            "authority_cardinality": 0,
            "manager_decision": construction,
            "execution_authority": "NOT_GRANTED",
            "independent_review": "REQUIRED_BEFORE_ANY_LATER_EXECUTION_AUTHORITY",
        },
        "execution_limit": 0,
        "retry": "FORBIDDEN",
        "replay": "FORBIDDEN",
        "resume": "FORBIDDEN",
        "relaunch": "FORBIDDEN",
        "identity": core.ONE_USE_IDENTITY,
        "argv": core.authorized_launch_argv(),
        "endpoint_argv": core.authorized_endpoint_argv(),
        "input_utf8_path": str(core.PACKAGE / "input.txt"),
        "generation_budget": core.GENERATION_BUDGET,
        "decoding": "official-tokenizer-whole-sequence-exactly-once",
        "transaction_nonce": core.TRANSACTION_NONCE,
        "completion_directory": str(core.COMPLETION),
        "bindings": {
            "package_sources": source_hashes,
            "package_checks_sha256": core.sha256_file(output / "package-checks.json"),
            "input_utf8_sha256": core.sha256_file(core.PACKAGE / "input.txt"),
            "descriptor_sha256": core.sha256_file(
                output / core.DESCRIPTOR.relative_to(core.PACKAGE)
            ),
            "endpoint_build_provenance_sha256": core.sha256_file(
                output / core.ENDPOINT_BUILD_PROVENANCE.relative_to(core.PACKAGE)
            ),
            "command_memory_sha256": core.sha256_file(
                output / core.COMMAND_MEMORY.relative_to(core.PACKAGE)
            ),
            "memory_image_sha256": core.sha256_file(
                output / core.MEMORY_IMAGE.relative_to(core.PACKAGE)
            ),
            "endpoint_binary_sha256": core.sha256_file(
                output / core.PRODUCT_BINARY.name
            ),
            "verilated_runtime_object_sha256": core.sha256_file(
                output / core.VERILATED_RUNTIME_OBJECT.relative_to(core.PACKAGE)
            ),
            "verilated_rtl_archive_sha256": core.sha256_file(
                output / core.VERILATED_RTL_ARCHIVE.relative_to(core.PACKAGE)
            ),
            "rtl_source_closure": core.rtl_source_closure(),
            "public_ace2_shell_interface": core.public_ace2_shell_interface(),
            "normalized_final_export_manifest_sha256": core.sha256_file(
                output / core.NORMALIZED_MANIFEST.relative_to(core.PACKAGE)
            ),
            "official_model_sha256": core.sha256_file(core.MODEL.resolve()),
            "reproducibility_report_sha256": core.sha256_file(
                output / core.REPRODUCIBILITY_REPORT.name
            ),
            "checkpoint_sha256": core.EXPECTED["checkpoint"],
            "export_manifest_sha256": core.EXPECTED["export_manifest"],
            "packed_w4_sha256": core.EXPECTED["packed_w4"],
            "weight_scale32_sha256": core.EXPECTED["weight_scale32"],
            "ordered_rows_sha256": core.EXPECTED["ordered_rows"],
            "per_row_digest_ledger_sha256": core.EXPECTED["row_digests"],
            "tokenizer_json_sha256": core.EXPECTED["tokenizer_json"],
            "tokenizer_config_sha256": core.EXPECTED["tokenizer_config"],
            "chat_template_sha256": core.sha256_bytes(chat_template.encode("utf-8")),
            "qualification_preflight_sha256": qualification["preflight_sha256"],
            "accepted_interfaces": qualification["accepted_interfaces"],
            "accepted_tokenizer": qualification["accepted_tokenizer"],
        },
        "accelerator_agreement": "NOT_EXECUTED",
        "readable_dialogue": "NOT_EXECUTED",
        "latency": "NOT_MEASURED",
        "latency_scopes": {
            "host_model_orchestration": "NOT_MEASURED",
            "endpoint_process_wall": "NOT_MEASURED",
            "simulator_cycles": "NOT_EXECUTED",
            "accelerator_latency": "NOT_MEASURED",
        },
        "software_fallback": False,
        "stage1_state": "OPEN",
        "stage1_closed": False,
        "stage2": "FORBIDDEN",
    }
    core.write_json_exclusive(output / core.AUTHORIZATION.name, authorization)


'''
    text = replace_region(
        text,
        "def write_authorization(",
        "def main()",
        writer,
        "authorization writer",
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
    def __init__(self, decoded: str = "test", error: Exception | None = None) -> None:
        self.decoded = decoded
        self.error = error
        self.decode_calls: list[list[int]] = []
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
        return core.VOCABULARY_SIZE

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.forms.get(token_id, f"token-{token_id}")

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
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

    def test_valid_ascii_and_official_whole_sequence_decode_once(self) -> None:
        tokenizer = FakeTokenizer("ASCII")
        decoded, artifact = self.decode(tokenizer, [1, 1, 1, 1])
        self.assertEqual(decoded, "ASCII")
        self.assertEqual(tokenizer.decode_calls, [[1, 1, 1, 1]])
        self.assertEqual(
            artifact["decode"]["classification"], "PASS_READABLE_STRICT_UTF8"
        )

    def test_multibyte_unicode_split_across_byte_fallback_pieces(self) -> None:
        tokenizer = FakeTokenizer("€ value")
        decoded, artifact = self.decode(tokenizer, [2, 3, 4, 151645])
        self.assertEqual(decoded, "€ value")
        self.assertEqual(tokenizer.decode_calls, [[2, 3, 4, 151645]])
        fragments = [step["byte_fragment"] for step in artifact["steps"][:3]]
        self.assertTrue(all(item["source"] == "BYTE_FALLBACK_TOKEN" for item in fragments))
        self.assertTrue(all(item["used_as_final_text"] is False for item in fragments))
        self.assertEqual(
            fragments[0]["partial_utf8_status"],
            "PARTIAL_FRAGMENT_NOT_READABLE_STRICT_UTF8_DIAGNOSTIC_ONLY",
        )

    def test_replacement_character_is_rejected_with_complete_artifact(self) -> None:
        tokenizer = FakeTokenizer("\ufffd")
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "predecode.json"
            with self.assertRaisesRegex(core.ProductError, "REPLACEMENT_CHARACTER"):
                driver.decode_token_ids_once(tokenizer, [1, 1, 1, 1], path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["status"], "FAIL_CLOSED")
            self.assertEqual(
                artifact["decode"]["classification"], "REPLACEMENT_CHARACTER_REJECTED"
            )
            self.assertEqual(artifact["canonical_generation_result"]["token_count"], 4)

    def test_invalid_and_out_of_range_ids_fail_before_decode(self) -> None:
        for token_ids in ([1, -1, 1, 1], [1, core.VOCABULARY_SIZE, 1, 1]):
            tokenizer = FakeTokenizer()
            with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
                path = Path(temporary) / "predecode.json"
                with self.assertRaisesRegex(core.ProductError, "outside vocabulary"):
                    driver.decode_token_ids_once(tokenizer, token_ids, path)
                self.assertEqual(tokenizer.decode_calls, [])
                artifact = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    artifact["decode"]["classification"],
                    "INVALID_GENERATION_TOKEN_ID_SEQUENCE",
                )

    def test_eos_and_four_token_generation_bookkeeping(self) -> None:
        tokenizer = FakeTokenizer("done")
        _, artifact = self.decode(tokenizer, [5, core.EOS_TOKEN_IDS.__iter__().__next__()])
        self.assertEqual(
            artifact["canonical_generation_result"]["stop_reason"], "eos"
        )
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            with self.assertRaisesRegex(core.ProductError, "does not contain four"):
                driver.decode_token_ids_once(
                    FakeTokenizer(), [1, 1, 1], Path(temporary) / "short.json"
                )

    def test_arbitrary_utf8_input_and_readability(self) -> None:
        value = "Grüße 世界 — registers"
        self.assertEqual(value.encode("utf-8").decode("utf-8", errors="strict"), value)
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

    def test_decode_exception_is_classified_without_success_result(self) -> None:
        tokenizer = FakeTokenizer(error=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"))
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            path = Path(temporary) / "predecode.json"
            with self.assertRaisesRegex(core.ProductError, "official whole-sequence"):
                driver.decode_token_ids_once(tokenizer, [1, 1, 1, 1], path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["decode"]["classification"],
                "OFFICIAL_WHOLE_SEQUENCE_DECODE_EXCEPTION",
            )
            self.assertFalse((Path(temporary) / "product-result.json").exists())

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
            "official_model_sha256": core.EXPECTED["model"],
            "rtl_source_closure": core.rtl_source_closure(),
            "public_ace2_shell_interface": core.public_ace2_shell_interface(),
        }
        cases = {
            "model": "official_model_sha256",
            "tokenizer": "tokenizer_json_sha256",
            "export": "export_manifest_sha256",
            "rtl": "rtl_source_closure",
        }
        for label, field in cases.items():
            with self.subTest(label=label):
                mutated = copy.deepcopy(bindings)
                mutated[field] = {} if field == "rtl_source_closure" else "0" * 64
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
            core.validate_package_source_bindings({"package_sources": source_bindings})

    def test_cf08_terminal_failure_and_zero_authority_are_authenticated(self) -> None:
        terminal = core.validate_cf08_terminal_failure()
        self.assertEqual(terminal["exit_code"], 2)
        self.assertEqual(terminal["elapsed_seconds"], 1363.5)
        self.assertTrue(terminal["authority_consumed"])
        self.assertEqual(core.validate_construction_only_record()["authority_cardinality"], 0)
        with self.assertRaisesRegex(core.ProductError, "no execution authority"):
            core.validate_authorization()

    def test_live_rtl_interface_provenance_is_exact(self) -> None:
        closure = core.rtl_source_closure()
        interface = core.public_ace2_shell_interface()
        self.assertEqual(len(closure["files"]), 19)
        self.assertEqual(interface["parameter_count"], 14)
        self.assertEqual(interface["port_count"], 64)

    def test_prepared_package_and_two_build_reproducibility(self) -> None:
        if not core.AUTHORIZATION.exists():
            self.skipTest("package is not materialized yet")
        authorization = core.validate_prepared_authorization()
        self.assertEqual(authorization["authorization_state"], "NOT_GRANTED")
        report = core.load_object(core.REPRODUCIBILITY_REPORT)
        self.assertEqual(report["status"], "PASS_TWO_BUILD_BYTE_MODE_EQUAL")
        self.assertEqual(report["build_count"], 2)
        core.assert_fresh_execution_state()
        for path in (
            core.ATTEMPT,
            core.CONSUMED,
            core.PACKAGE / "execution-registry.json",
            core.PACKAGE / "completion",
            core.PACKAGE / "product-result.json",
        ):
            self.assertFalse(path.exists(), str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


def main() -> int:
    if PACKAGE.exists():
        raise RuntimeError(f"fresh CF09 namespace already exists: {PACKAGE}")
    for required in (
        SOURCE / "package_core.py",
        SOURCE / "prepare_package.py",
        SOURCE / "product_driver.py",
        SOURCE / "launch.sh",
        SOURCE / "input.txt",
        TERMINAL_HANDOFF / "latest.json",
        TERMINAL_HANDOFF / "round-0002.json",
        TERMINAL_SEAL,
    ):
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(f"immutable prerequisite is absent or linked: {required}")
    PACKAGE.mkdir(parents=True)
    (PACKAGE / "authority").mkdir()
    (PACKAGE / "tests").mkdir()
    (PACKAGE / "package_core.py").write_text(make_core(), encoding="utf-8")
    (PACKAGE / "prepare_package.py").write_text(make_prepare(), encoding="utf-8")
    (PACKAGE / "product_driver.py").write_text(make_driver(), encoding="utf-8")
    (PACKAGE / "tests/test_product_package.py").write_text(TESTS, encoding="utf-8")
    shutil.copyfile(SOURCE / "input.txt", PACKAGE / "input.txt")
    shutil.copyfile(SOURCE / "launch.sh", PACKAGE / "launch.sh")
    manager_record = {
        "schema": "ace2-r6-external-manager-authority-v1",
        "status": "GRANTED",
        "authority_cardinality": 1,
        "semantic_fields": {
            "decision_id": CONSTRUCTION_DECISION,
            "scope": "r6decodeobservabilityprep09",
            "authorized_action": "CONSTRUCTION_AND_INDEPENDENT_REVIEW_ONLY",
            "sealed_cf07_identity": "stage1-w4a8-r6-product-prep-cf07-attempt-0001",
            "sealed_cf08_identity": "stage1-w4a8-r6-product-prep-cf08-attempt-0001",
            "sealed_cf08_outcome": "SEALED_RUNTIME_FAILURE",
            "sealed_cf08_retry": "FORBIDDEN",
            "sealed_cf08_replay": "FORBIDDEN",
            "sealed_cf08_resume": "FORBIDDEN",
            "sealed_cf08_relaunch": "FORBIDDEN",
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
            "independent_l2_review": "REQUIRED",
            "stage_transition": "DISABLED",
        },
    }
    (PACKAGE / "authority/manager-decision.json").write_bytes(
        canonical_bytes(manager_record)
    )
    modes = {
        "package_core.py": 0o444,
        "prepare_package.py": 0o555,
        "product_driver.py": 0o444,
        "tests/test_product_package.py": 0o444,
        "input.txt": 0o444,
        "launch.sh": 0o555,
        "authority/manager-decision.json": 0o444,
    }
    for name, mode in modes.items():
        (PACKAGE / name).chmod(mode)
    print(PACKAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
