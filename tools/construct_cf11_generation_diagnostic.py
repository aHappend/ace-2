#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf11"
CF10 = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf10"
TERMINAL_SEAL = CF10 / "terminal-seal.json"
TERMINAL_OBSERVABILITY = CF10 / "attempt-0001/predecode-observability.json"
TERMINAL_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6stage1terminalseal10"
)
IDENTITY = "stage1-w4a8-r6-product-prep-cf11-attempt-0001"
NONCE = "e10d588ac86e8cd5aa5b56393bee3e4adb4ce67db1f173ee5811efd09d086cd8"
CONSTRUCTION_AUTHORITY = "mgr-r6-cf11-generation-quality-diagnosis-01"
CF10_IDS = [1019, 1019, 198, 1019]
PREDECESSORS = {
    f"cf{number:02d}": (
        ROOT
        / f"build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf{number:02d}"
    )
    for number in range(7, 11)
}
MODEL_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TOKENIZER_JSON_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
TOKENIZER_MAPPING_SHA256 = (
    "e3060fc6c97520b4bc5bcfaab0fbfcde72bbc33d45b54ed3929798061cb34be7"
)
TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TOKENIZER_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / TOKENIZER_REVISION
)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def namespace_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"immutable namespace is absent or linked: {path}")
    entries: dict[str, dict[str, object]] = {}
    for entry in sorted(path.rglob("*")):
        relative = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            raise RuntimeError(f"immutable namespace contains a link: {entry}")
        mode = entry.stat().st_mode & 0o777
        if entry.is_dir():
            entries[relative] = {"kind": "directory", "mode": mode}
        elif entry.is_file():
            entries[relative] = {
                "kind": "file",
                "mode": mode,
                "size": entry.stat().st_size,
                "sha256": sha256_file(entry),
            }
        else:
            raise RuntimeError(f"unsupported namespace entry: {entry}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "entry_count": len(entries),
        "entries": entries,
    }


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def validate_terminal_gate() -> tuple[dict[str, Any], dict[str, Any]]:
    latest = load_object(TERMINAL_REVIEW / "latest.json")
    if latest.get("kind") != "handoff_ref":
        raise RuntimeError("CF10 terminal latest index is not a reviewed handoff")
    handoff_path = Path(latest.get("handoff", {}).get("path", ""))
    handoff = load_object(handoff_path)
    if (
        handoff.get("mission_id") != "r6stage1terminalseal10"
        or handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise RuntimeError("CF10 terminal seal lacks independent completion")

    seal = load_object(TERMINAL_SEAL)
    expected = {
        "status": "SEALED_DECODE_FAILURE",
        "classification": "DECODE_FAILURE",
        "outcome": "product_not_completed",
        "first_natural_terminal": True,
        "durable_process_starts": 1,
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise RuntimeError("CF10 terminal seal summary differs")
    if (
        seal.get("identity", {}).get("package_attempt")
        != "stage1-w4a8-r6-product-prep-cf10-attempt-0001"
        or seal.get("terminal", {}).get("exit_code") != 2
        or seal.get("process", {}).get("durable_elapsed_seconds") != 1034.3
        or seal.get("generation", {}).get("token_ids") != CF10_IDS
        or seal.get("generation", {}).get("stop_reason") != "max_new_tokens"
        or seal.get("decode_domain", {}).get("token_classifications")
        != ["MAPPED_BASE_TOKEN_ID"] * 4
        or seal.get("decode_domain", {}).get(
            "official_whole_sequence_decode_calls"
        )
        != 1
        or seal.get("decode_domain", {}).get("classification")
        != "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8"
        or seal.get("execution_artifacts", {})
        .get("endpoint_invocation", {})
        .get("status")
        != "ABSENT"
        or seal.get("rtl_provenance", {}).get("rtl_source_file_count") != 19
    ):
        raise RuntimeError("CF10 sole terminal evidence differs")
    observability = load_object(TERMINAL_OBSERVABILITY)
    if (
        observability.get("canonical_generation_result", {}).get("token_ids")
        != CF10_IDS
        or observability.get("decode", {}).get("mode", {}).get("call_count") != 1
        or observability.get("decode", {}).get("classification")
        != "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8"
        or [
            step.get("tokenizer_id_classification")
            for step in observability.get("steps", [])
        ]
        != ["MAPPED_BASE_TOKEN_ID"] * 4
    ):
        raise RuntimeError("CF10 predecode observability differs")
    return handoff, seal


def readable_text(text: Any) -> bool:
    return (
        isinstance(text, str)
        and bool(text.strip())
        and "\ufffd" not in text
        and any(character.isalnum() for character in text)
        and not any(
            ord(character) < 32 and character not in "\n\t" for character in text
        )
    )


def reproduce_decode_once() -> dict[str, Any]:
    tokenizer_json = TOKENIZER_ROOT / "tokenizer.json"
    tokenizer_config = TOKENIZER_ROOT / "tokenizer_config.json"
    model = TOKENIZER_ROOT / "model.safetensors"
    expected_files = {
        tokenizer_json: TOKENIZER_JSON_SHA256,
        tokenizer_config: TOKENIZER_CONFIG_SHA256,
        model: MODEL_SHA256,
    }
    for path, expected in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"authenticated tokenizer prerequisite differs: {path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ROOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    mapping = tokenizer.get_vocab()
    mapping_sha256 = hashlib.sha256(
        canonical_bytes(sorted(mapping.items(), key=lambda item: item[0]))
    ).hexdigest()
    if (
        tokenizer.vocab_size != 151643
        or len(tokenizer) != 151665
        or len(mapping) != 151665
        or mapping_sha256 != TOKENIZER_MAPPING_SHA256
    ):
        raise RuntimeError("authenticated official tokenizer mapping differs")

    decoded = tokenizer.decode(
        CF10_IDS,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise RuntimeError("official tokenizer returned non-text")
    raw = decoded.encode("utf-8", errors="strict")
    classification = (
        "PASS_READABLE_STRICT_UTF8"
        if readable_text(decoded)
        else "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8"
    )
    if classification != "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8":
        raise RuntimeError("official CF10 decode no longer reproduces the terminal")
    return {
        "schema": "ace2-r6-cf11-cf10-decode-reproduction-v1",
        "status": "PASS_READ_ONLY_OFFICIAL_DECODE_REPRODUCED",
        "source": {
            "terminal_seal_sha256": sha256_file(TERMINAL_SEAL),
            "predecode_observability_sha256": sha256_file(
                TERMINAL_OBSERVABILITY
            ),
            "token_ids": CF10_IDS,
        },
        "tokenizer": {
            "revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "mapping_sha256": TOKENIZER_MAPPING_SHA256,
            "base_vocabulary_size": 151643,
            "mapped_id_count": 151665,
        },
        "decode": {
            "api": "tokenizer.decode",
            "whole_sequence": True,
            "call_count": 1,
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
            "decoded_text": decoded,
            "utf8_bytes_hex": raw.hex(),
            "escaped_utf8_bytes": "".join(f"\\x{value:02x}" for value in raw),
            "escaped_code_units": decoded.encode(
                "unicode_escape", errors="strict"
            ).decode("ascii"),
            "code_points": [
                {
                    "index": index,
                    "value": f"U+{ord(character):04X}",
                    "escaped": character.encode("unicode_escape").decode("ascii"),
                }
                for index, character in enumerate(decoded)
            ],
            "classification": classification,
            "readable_text_policy_unchanged": True,
        },
        "execution": {
            "model_generation": "NOT_EXECUTED",
            "product_endpoint": "NOT_EXECUTED",
            "rtl": "NOT_EXECUTED",
        },
    }


CORE_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
IDENTITY = "stage1-w4a8-r6-product-prep-cf11-attempt-0001"
TRANSACTION_NONCE = "e10d588ac86e8cd5aa5b56393bee3e4adb4ce67db1f173ee5811efd09d086cd8"
MODEL_VOCABULARY_SIZE = 151936
TOKENIZER_BASE_VOCABULARY_SIZE = 151643
TOKENIZER_MAPPED_ID_COUNT = 151665
TOKENIZER_ADDED_TOKEN_COUNT = 22
TOKENIZER_MAXIMUM_MAPPED_ID = 151664
GENERATION_BUDGET = 4
FAILURE_STAGES = (
    "PROMPT_CHAT_TEMPLATE_SERIALIZATION",
    "EMBEDDING_INPUT_IDS",
    "PER_LAYER_STATE",
    "KV_STATE",
    "RTL_LOGIT_TILES",
    "TIED_LM_HEAD_LOGITS",
    "RANK_SELECTION",
    "CACHE_PLUMBING",
    "HOST_DECODE",
)


class DiagnosticError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON root is not an object: {path}")
    return value


def readable_text(text: Any) -> bool:
    return (
        isinstance(text, str)
        and bool(text.strip())
        and "\ufffd" not in text
        and any(character.isalnum() for character in text)
        and not any(
            ord(character) < 32 and character not in "\n\t" for character in text
        )
    )


def classify_decoded_text(text: Any) -> str:
    if not isinstance(text, str):
        return "OFFICIAL_DECODE_RETURNED_NON_TEXT"
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return "OFFICIAL_DECODE_NOT_STRICT_UTF8"
    if "\ufffd" in text:
        return "REPLACEMENT_CHARACTER_REJECTED"
    if not readable_text(text):
        return "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8"
    return "PASS_READABLE_STRICT_UTF8"


def validate_static_contract() -> dict[str, Any]:
    contract = load_object(PACKAGE / "package-contract.json")
    spec = load_object(PACKAGE / "future-probe-spec.json")
    reference_requirements = spec.get("independent_reference_requirements", {})
    producer = reference_requirements.get("authorized_reference_producer")
    attester = reference_requirements.get("execution_attestation_authority")
    producer_manifest_path = PACKAGE / "reference-producer-manifest.json"
    producer_source_path = PACKAGE / "independent_reference_producer.py"
    if (
        contract.get("identity") != IDENTITY
        or contract.get("transaction_nonce") != TRANSACTION_NONCE
        or contract.get("execution_limit") != 0
        or contract.get("product_execution") != "FORBIDDEN"
        or contract.get("model_generation") != "FORBIDDEN"
        or contract.get("rtl_execution") != "FORBIDDEN"
        or contract.get("independent_l2_review") != "REQUIRED_BEFORE_ANY_AUTHORITY"
        or contract.get("stage1_state") != "OPEN"
        or contract.get("stage2") != "FORBIDDEN"
    ):
        raise DiagnosticError("CF11 static contract differs")
    if contract.get("future_probe_spec_sha256") != sha256_file(
        PACKAGE / "future-probe-spec.json"
    ):
        raise DiagnosticError("future probe specification hash differs")
    if spec.get("expected_tokens") is not None:
        raise DiagnosticError("future probe invents expected tokens")
    if (
        not isinstance(attester, dict)
        or attester.get("status") != "REQUIRED_NOT_BOUND"
        or attester.get("algorithm") != "Ed25519"
        or any(
            attester.get(field) is not None
            for field in (
                "key_id",
                "verification_key_base64",
                "verification_key_sha256",
            )
        )
    ):
        raise DiagnosticError("reference execution attester was prematurely authorized")
    if (
        not isinstance(producer, dict)
        or producer.get("implementation_sha256")
        != sha256_file(producer_source_path)
        or producer.get("source_manifest_sha256")
        != sha256_file(producer_manifest_path)
    ):
        raise DiagnosticError("authorized reference producer binding differs")
    producer_manifest = load_object(producer_manifest_path)
    if (
        producer_manifest.get("implementation", {}).get("sha256")
        != producer["implementation_sha256"]
        or producer_manifest.get("activation_gate")
        != "CURRENT_CF11_INDEPENDENT_L2_REVIEW_DONE"
        or producer_manifest.get("execution_authority") != "NOT_GRANTED"
    ):
        raise DiagnosticError("reference producer manifest differs")
    return contract
'''


PROBE_SOURCE = r'''
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

import diagnostic_core as core


def _atomic_json(path: Path, value: object, *, exclusive: bool) -> None:
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


def _digest_ids(token_ids: list[int]) -> str:
    return hashlib.sha256(core.canonical_bytes(token_ids)).hexdigest()


def _valid_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_top_k(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise core.DiagnosticError(f"{label} top-k is incomplete")
    prior_logit: float | None = None
    seen: set[int] = set()
    for rank, item in enumerate(value, 1):
        if (
            not isinstance(item, dict)
            or type(item.get("token_id")) is not int
            or not 0 <= item["token_id"] < core.MODEL_VOCABULARY_SIZE
            or not isinstance(item.get("logit"), (int, float))
            or isinstance(item.get("logit"), bool)
            or item.get("rank") != rank
            or item["token_id"] in seen
        ):
            raise core.DiagnosticError(f"{label} top-k entry differs")
        logit = float(item["logit"])
        if prior_logit is not None and logit > prior_logit:
            raise core.DiagnosticError(f"{label} top-k ordering differs")
        prior_logit = logit
        seen.add(item["token_id"])
    return value


REFERENCE_POSITION_FIELDS = {
    "expected_argmax",
    "top_k",
    "prompt_bytes_sha256",
    "chat_template_token_ids_sha256",
    "embedding_input_ids_sha256",
    "per_layer_state_digests",
    "per_layer_kv_digests",
    "logits_sha256",
}


def _decode_base64(value: object, label: str) -> bytes:
    if not isinstance(value, str):
        raise core.DiagnosticError(f"{label} is not base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise core.DiagnosticError(f"{label} is not canonical base64") from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise core.DiagnosticError(f"{label} is not canonical base64")
    return decoded


def _validate_execution_attestation(
    path: Path | None,
    artifact: dict[str, Any],
    artifact_sha256: str,
    spec: dict[str, Any],
) -> str:
    authority = spec.get("independent_reference_requirements", {}).get(
        "execution_attestation_authority"
    )
    if (
        not isinstance(authority, dict)
        or authority.get("status") != "BOUND_BY_FUTURE_EXECUTION_AUTHORITY"
        or authority.get("algorithm") != "Ed25519"
        or not isinstance(authority.get("key_id"), str)
        or not authority["key_id"]
        or not _valid_digest(authority.get("verification_key_sha256"))
        or path is None
    ):
        raise core.DiagnosticError(
            "independent reference execution attestation is required and not authorized"
        )
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise core.DiagnosticError(
            "independent reference execution attestation is absent, linked, or mutable"
        )
    attestation = core.load_object(path)
    if path.read_bytes() != core.canonical_bytes(attestation):
        raise core.DiagnosticError(
            "independent reference execution attestation is not canonical"
        )
    signer = attestation.get("signer")
    payload = attestation.get("payload")
    signed = {
        field: attestation.get(field)
        for field in ("schema", "status", "trust_domain", "signer", "payload")
    }
    if (
        set(attestation)
        != {
            "schema",
            "status",
            "trust_domain",
            "signer",
            "payload",
            "signature_base64",
        }
        or attestation.get("schema")
        != "ace2-r6-cf11-independent-reference-execution-attestation-v1"
        or attestation.get("status")
        != "PASS_REFERENCE_PRODUCER_EXECUTED_BEFORE_PRODUCT"
        or attestation.get("trust_domain")
        != "EXTERNAL_FUTURE_EXECUTION_ATTESTER"
        or not isinstance(signer, dict)
        or set(signer)
        != {
            "algorithm",
            "key_id",
            "verification_key_sha256",
            "private_key_access",
        }
        or signer.get("algorithm") != authority["algorithm"]
        or signer.get("key_id") != authority["key_id"]
        or signer.get("verification_key_sha256")
        != authority["verification_key_sha256"]
        or signer.get("private_key_access")
        != "INACCESSIBLE_TO_PRODUCT_AND_REFERENCE_ARTIFACT_CODE"
        or not isinstance(payload, dict)
        or set(payload)
        != {
            "reference_artifact_sha256",
            "producer_implementation_sha256",
            "producer_source_manifest_sha256",
            "future_probe_spec_sha256",
            "model_sha256",
            "tokenizer_json_sha256",
            "process_argv_sha256",
            "process_exit_code",
            "product_execution_started",
            "product_positions_consumed",
        }
        or payload.get("reference_artifact_sha256") != artifact_sha256
        or payload.get("producer_implementation_sha256")
        != artifact["producer"]["implementation_sha256"]
        or payload.get("producer_source_manifest_sha256")
        != artifact["producer"]["source_manifest_sha256"]
        or payload.get("future_probe_spec_sha256")
        != core.sha256_file(core.PACKAGE / "future-probe-spec.json")
        or payload.get("model_sha256") != spec["bindings"]["official_model_sha256"]
        or payload.get("tokenizer_json_sha256")
        != spec["bindings"]["tokenizer_json_sha256"]
        or not _valid_digest(payload.get("process_argv_sha256"))
        or payload.get("process_exit_code") != 0
        or payload.get("product_execution_started") is not False
        or payload.get("product_positions_consumed") is not False
    ):
        raise core.DiagnosticError(
            "independent reference execution attestation binding differs"
        )
    public_key_raw = _decode_base64(
        authority.get("verification_key_base64"), "attestation verification key"
    )
    if (
        len(public_key_raw) != 32
        or hashlib.sha256(public_key_raw).hexdigest()
        != authority["verification_key_sha256"]
    ):
        raise core.DiagnosticError(
            "independent reference attestation verification key differs"
        )
    signature = _decode_base64(
        attestation.get("signature_base64"), "reference execution signature"
    )
    if len(signature) != 64:
        raise core.DiagnosticError("reference execution signature length differs")
    try:
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
            signature, core.canonical_bytes(signed)
        )
    except (InvalidSignature, ValueError) as error:
        raise core.DiagnosticError(
            "independent reference execution signature is invalid"
        ) from error
    return core.sha256_file(path)


def _load_reference_artifact(
    path: Path, spec: dict[str, Any], attestation_path: Path | None
) -> tuple[dict[str, Any], str, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise core.DiagnosticError(
            "independent reference artifact is absent, linked, or mutable"
        )
    artifact = core.load_object(path)
    if path.read_bytes() != core.canonical_bytes(artifact):
        raise core.DiagnosticError("independent reference artifact is not canonical")
    producer = artifact.get("producer")
    authorized_producer = spec.get("independent_reference_requirements", {}).get(
        "authorized_reference_producer"
    )
    if (
        set(artifact)
        != {
            "schema",
            "status",
            "producer",
            "probe_spec_sha256",
            "model_sha256",
            "tokenizer_json_sha256",
            "full_logit_count",
            "positions",
        }
        or artifact.get("schema")
        != "ace2-r6-cf11-independent-reference-artifact-v1"
        or artifact.get("status") != "SEALED_BEFORE_PRODUCT_EXECUTION"
        or not isinstance(producer, dict)
        or set(producer)
        != {
            "kind",
            "independent_of_product",
            "execution_domain",
            "implementation_sha256",
            "source_manifest_sha256",
        }
        or producer.get("kind") != "INDEPENDENT_HOST_REFERENCE"
        or producer.get("independent_of_product") is not True
        or producer.get("execution_domain")
        != "SEPARATE_HOST_REFERENCE_PROCESS"
        or not _valid_digest(producer.get("implementation_sha256"))
        or not _valid_digest(producer.get("source_manifest_sha256"))
        or producer != authorized_producer
        or artifact.get("probe_spec_sha256")
        != core.sha256_file(core.PACKAGE / "future-probe-spec.json")
        or artifact.get("model_sha256")
        != spec["bindings"]["official_model_sha256"]
        or artifact.get("tokenizer_json_sha256")
        != spec["bindings"]["tokenizer_json_sha256"]
        or artifact.get("full_logit_count") != core.MODEL_VOCABULARY_SIZE
        or not isinstance(artifact.get("positions"), list)
        or len(artifact["positions"]) != core.GENERATION_BUDGET
        or any(
            not isinstance(position, dict)
            or set(position) != REFERENCE_POSITION_FIELDS
            for position in artifact["positions"]
        )
    ):
        raise core.DiagnosticError(
            "independent reference artifact producer is not authorized by the "
            "future-probe contract"
        )
    artifact_sha256 = core.sha256_file(path)
    attestation_sha256 = _validate_execution_attestation(
        attestation_path, artifact, artifact_sha256, spec
    )
    return artifact, artifact_sha256, attestation_sha256


def _validate_reference(
    reference: dict[str, Any],
    product: dict[str, Any],
    spec: dict[str, Any],
    ordinal: int,
    artifact: dict[str, Any],
    artifact_sha256: str,
) -> None:
    bindings = spec["bindings"]
    payload = {field: reference.get(field) for field in REFERENCE_POSITION_FIELDS}
    producer = artifact["producer"]
    if (
        set(reference) != REFERENCE_POSITION_FIELDS | {"source_artifact_sha256"}
        or reference.get("source_artifact_sha256") != artifact_sha256
        or payload != artifact["positions"][ordinal]
        or artifact.get("model_sha256") != bindings["official_model_sha256"]
        or artifact.get("tokenizer_json_sha256")
        != bindings["tokenizer_json_sha256"]
        or not _valid_digest(reference.get("logits_sha256"))
    ):
        raise core.DiagnosticError(
            "reference does not match its immutable independent artifact"
        )
    if (
        product.get("source_artifact_sha256") == artifact_sha256
        or product.get("producer_implementation_sha256")
        == producer["implementation_sha256"]
        or product.get("producer_source_manifest_sha256")
        == producer["source_manifest_sha256"]
    ):
        raise core.DiagnosticError(
            "reference and product provenance are not independent"
        )
    top_k = _validate_top_k(reference.get("top_k"), "reference")
    if reference.get("expected_argmax") != top_k[0]["token_id"]:
        raise core.DiagnosticError("reference argmax is not independently derived")


def _validate_digest_list(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != 24
        or not all(_valid_digest(item) for item in value)
    ):
        raise core.DiagnosticError(f"{label} lacks 24 authenticated digests")
    return value


def _first_divergence(position: dict[str, Any]) -> dict[str, Any]:
    reference = position["reference"]
    product = position["product"]
    rtl = position["rtl"]
    cache = position["cache_comparison"]
    comparisons = (
        (
            "PROMPT_CHAT_TEMPLATE_SERIALIZATION",
            reference["prompt_bytes_sha256"],
            product["prompt_bytes_sha256"],
        ),
        (
            "PROMPT_CHAT_TEMPLATE_SERIALIZATION",
            reference["chat_template_token_ids_sha256"],
            product["chat_template_token_ids_sha256"],
        ),
        (
            "EMBEDDING_INPUT_IDS",
            reference["embedding_input_ids_sha256"],
            product["embedding_input_ids_sha256"],
        ),
    )
    for stage, expected, observed in comparisons:
        if expected != observed:
            return {
                "status": "DIVERGED",
                "stage": stage,
                "classification": f"{stage}_DIGEST_MISMATCH",
            }
    for stage, field in (
        ("PER_LAYER_STATE", "per_layer_state_digests"),
        ("KV_STATE", "per_layer_kv_digests"),
    ):
        for layer, (expected, observed) in enumerate(
            zip(reference[field], product[field], strict=True)
        ):
            if expected != observed:
                return {
                    "status": "DIVERGED",
                    "stage": stage,
                    "layer": layer,
                    "classification": f"{stage}_DIGEST_MISMATCH",
                }
    if reference["logits_sha256"] != product["logits_sha256"]:
        return {
            "status": "DIVERGED",
            "stage": "TIED_LM_HEAD_LOGITS",
            "classification": "INDEPENDENT_REFERENCE_PRODUCT_LOGITS_MISMATCH",
        }
    if rtl["logits_sha256"] != product["logits_sha256"]:
        return {
            "status": "DIVERGED",
            "stage": "RTL_LOGIT_TILES",
            "classification": "RTL_PRODUCT_LOGITS_MISMATCH",
        }
    if reference["expected_argmax"] != position["selected_token_id"]:
        return {
            "status": "DIVERGED",
            "stage": "RANK_SELECTION",
            "classification": "SELECTED_TOKEN_DISAGREES_WITH_INDEPENDENT_ARGMAX",
        }
    if (
        cache["cache_free_logits_sha256"]
        != cache["cacheful_logits_sha256"]
        or cache["cache_free_selected_token_id"]
        != cache["cacheful_selected_token_id"]
    ):
        return {
            "status": "DIVERGED",
            "stage": "CACHE_PLUMBING",
            "classification": "CACHE_FREE_CACHEFUL_MISMATCH",
        }
    return {
        "status": "NO_DIVERGENCE_AT_RECORDED_BOUNDARIES",
        "stage": None,
        "classification": "BOUNDARIES_AGREE",
    }


def validate_position(
    position: dict[str, Any],
    spec: dict[str, Any],
    reference_artifact_path: Path,
    reference_attestation_path: Path | None,
) -> dict[str, Any]:
    if (
        type(position.get("ordinal")) is not int
        or not 0 <= position["ordinal"] < core.GENERATION_BUDGET
        or type(position.get("absolute_position")) is not int
        or type(position.get("selected_token_id")) is not int
        or not 0 <= position["selected_token_id"] < core.MODEL_VOCABULARY_SIZE
    ):
        raise core.DiagnosticError("generated position identity differs")
    reference = position.get("reference")
    product = position.get("product")
    rtl = position.get("rtl")
    cache = position.get("cache_comparison")
    if not all(isinstance(item, dict) for item in (reference, product, rtl, cache)):
        raise core.DiagnosticError("generated position boundary set is incomplete")
    artifact, artifact_sha256, _ = _load_reference_artifact(
        reference_artifact_path, spec, reference_attestation_path
    )
    _validate_reference(
        reference,
        product,
        spec,
        position["ordinal"],
        artifact,
        artifact_sha256,
    )
    for owner, record in (("reference", reference), ("product", product)):
        for field in (
            "prompt_bytes_sha256",
            "chat_template_token_ids_sha256",
            "embedding_input_ids_sha256",
            "logits_sha256",
        ):
            if not _valid_digest(record.get(field)):
                raise core.DiagnosticError(f"{owner} {field} differs")
        _validate_digest_list(record.get("per_layer_state_digests"), f"{owner} state")
        _validate_digest_list(record.get("per_layer_kv_digests"), f"{owner} KV")
        _validate_top_k(record.get("top_k"), owner)
    selected_entries = [
        item
        for item in product["top_k"]
        if item["token_id"] == position["selected_token_id"]
    ]
    if (
        type(product.get("selected_rank")) is not int
        or product["selected_rank"] < 1
        or not isinstance(product.get("selected_margin"), (int, float))
        or isinstance(product.get("selected_margin"), bool)
        or len(selected_entries) != 1
        or selected_entries[0]["rank"] != product["selected_rank"]
        or not _valid_digest(product.get("source_artifact_sha256"))
        or not _valid_digest(product.get("producer_implementation_sha256"))
        or not _valid_digest(product.get("producer_source_manifest_sha256"))
    ):
        raise core.DiagnosticError("selected token rank or margin differs")
    if (
        reference["logits_sha256"] == product["logits_sha256"]
        and reference["top_k"] != product["top_k"]
    ):
        raise core.DiagnosticError("equal reference/product logits have unequal top-k")
    if (
        rtl.get("source") != "AUTHENTICATED_RTL_OR_PRODUCT_LOGIT_TILES"
        or not _valid_digest(rtl.get("logits_sha256"))
        or not _valid_digest(rtl.get("source_artifact_sha256"))
    ):
        raise core.DiagnosticError("RTL/product logit evidence is unauthenticated")
    _validate_top_k(rtl.get("top_k"), "RTL/product")
    for field in (
        "cache_free_logits_sha256",
        "cacheful_logits_sha256",
    ):
        if not _valid_digest(cache.get(field)):
            raise core.DiagnosticError(f"{field} differs")
    for field in (
        "cache_free_selected_token_id",
        "cacheful_selected_token_id",
    ):
        if type(cache.get(field)) is not int:
            raise core.DiagnosticError(f"{field} differs")
    result = copy.deepcopy(position)
    result["first_divergence"] = _first_divergence(result)
    return result


class FutureProbeJournal:
    def __init__(
        self,
        path: Path,
        reference_artifact_path: Path,
        reference_attestation_path: Path,
    ) -> None:
        self.path = path
        self.reference_artifact_path = reference_artifact_path
        self.reference_attestation_path = reference_attestation_path

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        prompt_bytes: bytes,
        chat_template_token_ids: list[int],
        reference_artifact_path: Path,
        reference_attestation_path: Path,
    ) -> "FutureProbeJournal":
        prompt_text = prompt_bytes.decode("utf-8", errors="strict")
        if not chat_template_token_ids or not all(
            type(token_id) is int
            and 0 <= token_id < core.MODEL_VOCABULARY_SIZE
            for token_id in chat_template_token_ids
        ):
            raise core.DiagnosticError("chat-template token IDs differ")
        spec = core.load_object(core.PACKAGE / "future-probe-spec.json")
        reference_artifact, reference_artifact_sha256, attestation_sha256 = (
            _load_reference_artifact(
                reference_artifact_path, spec, reference_attestation_path
            )
        )
        prompt_sha256 = core.sha256_bytes(prompt_bytes)
        template_sha256 = _digest_ids(chat_template_token_ids)
        if any(
            position["prompt_bytes_sha256"] != prompt_sha256
            or position["chat_template_token_ids_sha256"] != template_sha256
            for position in reference_artifact["positions"]
        ):
            raise core.DiagnosticError(
                "reference artifact prompt or template binding differs"
            )
        artifact = {
            "schema": "ace2-r6-cf11-future-execution-evidence-v1",
            "status": "PROMPT_AND_TEMPLATE_ATOMICALLY_PUBLISHED",
            "identity": core.IDENTITY,
            "transaction_nonce": core.TRANSACTION_NONCE,
            "prompt": {
                "utf8_text": prompt_text,
                "bytes_hex": prompt_bytes.hex(),
                "escaped_utf8_bytes": "".join(
                    f"\\x{value:02x}" for value in prompt_bytes
                ),
                "byte_count": len(prompt_bytes),
                "sha256": core.sha256_bytes(prompt_bytes),
            },
            "chat_template_token_ids": list(chat_template_token_ids),
            "chat_template_token_ids_sha256": _digest_ids(
                chat_template_token_ids
            ),
            "independent_reference_artifact": {
                "sha256": reference_artifact_sha256,
                "producer_implementation_sha256": reference_artifact[
                    "producer"
                ]["implementation_sha256"],
                "producer_source_manifest_sha256": reference_artifact[
                    "producer"
                ]["source_manifest_sha256"],
                "execution_attestation_sha256": attestation_sha256,
                "sealed_before_product_execution": True,
            },
            "positions": [],
            "failures": [],
            "first_divergence": None,
            "decode": {
                "status": "NOT_STARTED",
                "whole_sequence": True,
                "call_count": 0,
            },
        }
        _atomic_json(path, artifact, exclusive=True)
        return cls(path, reference_artifact_path, reference_attestation_path)

    def _load(self) -> dict[str, Any]:
        return core.load_object(self.path)

    def record_position(self, position: dict[str, Any]) -> dict[str, Any]:
        artifact = self._load()
        if artifact["decode"]["status"] != "NOT_STARTED":
            raise core.DiagnosticError("position evidence arrived after decode")
        if position.get("ordinal") != len(artifact["positions"]):
            raise core.DiagnosticError("generated position ordering differs")
        spec = core.load_object(core.PACKAGE / "future-probe-spec.json")
        if (
            core.sha256_file(self.reference_artifact_path)
            != artifact["independent_reference_artifact"]["sha256"]
            or core.sha256_file(self.reference_attestation_path)
            != artifact["independent_reference_artifact"][
                "execution_attestation_sha256"
            ]
        ):
            raise core.DiagnosticError(
                "independent reference artifact or attestation changed after journal creation"
            )
        validated = validate_position(
            position,
            spec,
            self.reference_artifact_path,
            self.reference_attestation_path,
        )
        artifact["positions"].append(validated)
        divergence = validated["first_divergence"]
        if (
            artifact["first_divergence"] is None
            and divergence["status"] == "DIVERGED"
        ):
            artifact["first_divergence"] = {
                **divergence,
                "ordinal": validated["ordinal"],
                "absolute_position": validated["absolute_position"],
            }
        artifact["status"] = "POSITION_EVIDENCE_ATOMICALLY_PUBLISHED"
        _atomic_json(self.path, artifact, exclusive=False)
        return validated

    def record_failure(
        self, stage: str, classification: str, error_type: str
    ) -> None:
        if stage not in core.FAILURE_STAGES:
            raise core.DiagnosticError("failure stage is outside the probe schema")
        artifact = self._load()
        artifact["failures"].append(
            {
                "stage": stage,
                "classification": classification,
                "error_type": error_type,
                "positions_persisted": len(artifact["positions"]),
            }
        )
        artifact["status"] = "FAILURE_EVIDENCE_ATOMICALLY_PUBLISHED"
        if artifact["first_divergence"] is None:
            artifact["first_divergence"] = {
                "status": "FAILURE",
                "stage": stage,
                "classification": classification,
            }
        _atomic_json(self.path, artifact, exclusive=False)

    def mark_predecode_complete(self) -> None:
        artifact = self._load()
        if len(artifact["positions"]) != core.GENERATION_BUDGET:
            raise core.DiagnosticError(
                "decode forbidden until every generated position is persisted"
            )
        artifact["status"] = "ALL_REQUIRED_EVIDENCE_PERSISTED_BEFORE_DECODE"
        artifact["decode"]["status"] = "READY_FOR_ONE_OFFICIAL_WHOLE_SEQUENCE_CALL"
        _atomic_json(self.path, artifact, exclusive=False)

    def record_decode_result(self, decoded: object) -> None:
        artifact = self._load()
        if (
            artifact["decode"]["status"]
            != "READY_FOR_ONE_OFFICIAL_WHOLE_SEQUENCE_CALL"
        ):
            raise core.DiagnosticError("decode result arrived before evidence barrier")
        classification = core.classify_decoded_text(decoded)
        artifact["decode"] = {
            "status": (
                "PASS"
                if classification == "PASS_READABLE_STRICT_UTF8"
                else "FAIL_CLOSED"
            ),
            "whole_sequence": True,
            "call_count": 1,
            "classification": classification,
        }
        artifact["status"] = "DECODE_RESULT_ATOMICALLY_PUBLISHED"
        if (
            classification != "PASS_READABLE_STRICT_UTF8"
            and artifact["first_divergence"] is None
        ):
            artifact["first_divergence"] = {
                "status": "DIVERGED",
                "stage": "HOST_DECODE",
                "classification": classification,
            }
        _atomic_json(self.path, artifact, exclusive=False)
'''


REFERENCE_PRODUCER_SOURCE = r'''
#!/usr/bin/env python3
"""Pinned future-only independent host reference producer for CF11."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
GENERATION_BUDGET = 4
MODEL_VOCABULARY_SIZE = 151936
TOP_K_COUNT = 16


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(canonical_bytes({"dtype": str(value.dtype), "shape": list(value.shape)}))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def kv_digests(past_key_values: Any) -> list[str]:
    legacy = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else past_key_values
    )
    if not isinstance(legacy, (list, tuple)) or len(legacy) != 24:
        raise RuntimeError("independent reference lacks 24 KV layers")
    result = []
    for key, value in legacy:
        digest = hashlib.sha256()
        digest.update(tensor_digest(key).encode("ascii"))
        digest.update(tensor_digest(value).encode("ascii"))
        result.append(digest.hexdigest())
    return result


def ordered_top_k(logits: Any) -> list[dict[str, object]]:
    import torch

    order = torch.argsort(logits, descending=True, stable=True)[:TOP_K_COUNT]
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "logit": float(logits[token_id]),
        }
        for rank, token_id in enumerate(order.tolist(), 1)
    ]


def atomic_publish(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError("reference artifact output must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        path.chmod(0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def produce(model_dir: Path, prompt_bytes: bytes) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = json.loads((PACKAGE / "future-probe-spec.json").read_text(encoding="utf-8"))
    authorized = spec["independent_reference_requirements"][
        "authorized_reference_producer"
    ]
    if (
        sha256_file(Path(__file__)) != authorized["implementation_sha256"]
        or sha256_file(PACKAGE / "reference-producer-manifest.json")
        != authorized["source_manifest_sha256"]
    ):
        raise RuntimeError("reference producer source binding differs")
    bindings = spec["bindings"]
    if (
        sha256_file(model_dir / "model.safetensors")
        != bindings["official_model_sha256"]
        or sha256_file(model_dir / "tokenizer.json")
        != bindings["tokenizer_json_sha256"]
        or sha256_file(model_dir / "tokenizer_config.json")
        != bindings["tokenizer_config_sha256"]
    ):
        raise RuntimeError("reference producer model or tokenizer binding differs")

    prompt = prompt_bytes.decode("utf-8", errors="strict")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()
    messages = [{"role": "user", "content": prompt}]
    template = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    template_ids = [int(value) for value in template.reshape(-1).tolist()]
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    template_sha256 = hashlib.sha256(canonical_bytes(template_ids)).hexdigest()
    sequence = list(template_ids)
    positions = []
    past_key_values = None
    attention_mask = torch.ones((1, len(sequence)), dtype=torch.long)

    with torch.inference_mode():
        for ordinal in range(GENERATION_BUDGET):
            full_ids = torch.tensor([sequence], dtype=torch.long)
            cache_input = full_ids if ordinal == 0 else full_ids[:, -1:]
            cached = model(
                input_ids=cache_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            free = model(
                input_ids=full_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = free.logits[0, -1].detach().cpu().to(torch.float32).contiguous()
            if logits.numel() != MODEL_VOCABULARY_SIZE:
                raise RuntimeError("reference logits do not cover the model vocabulary")
            top_k = ordered_top_k(logits)
            expected = top_k[0]["token_id"]
            hidden_states = free.hidden_states[1:]
            if len(hidden_states) != 24:
                raise RuntimeError("independent reference lacks 24 layer states")
            positions.append(
                {
                    "expected_argmax": expected,
                    "top_k": top_k,
                    "prompt_bytes_sha256": prompt_sha256,
                    "chat_template_token_ids_sha256": template_sha256,
                    "embedding_input_ids_sha256": hashlib.sha256(
                        canonical_bytes(sequence)
                    ).hexdigest(),
                    "per_layer_state_digests": [
                        tensor_digest(state[:, -1]) for state in hidden_states
                    ],
                    "per_layer_kv_digests": kv_digests(cached.past_key_values),
                    "logits_sha256": tensor_digest(logits),
                }
            )
            sequence.append(int(expected))
            past_key_values = cached.past_key_values
            attention_mask = torch.ones((1, len(sequence)), dtype=torch.long)

    return {
        "schema": "ace2-r6-cf11-independent-reference-artifact-v1",
        "status": "SEALED_BEFORE_PRODUCT_EXECUTION",
        "producer": authorized,
        "probe_spec_sha256": sha256_file(PACKAGE / "future-probe-spec.json"),
        "model_sha256": bindings["official_model_sha256"],
        "tokenizer_json_sha256": bindings["tokenizer_json_sha256"],
        "full_logit_count": MODEL_VOCABULARY_SIZE,
        "positions": positions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_publish(args.output, produce(args.model, args.prompt_file.read_bytes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


TEST_SOURCE = r'''
from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import diagnostic_core as core
import future_probe as probe


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def top_k(first: int, second: int) -> list[dict[str, object]]:
    return [
        {"rank": 1, "token_id": first, "logit": 2.0},
        {"rank": 2, "token_id": second, "logit": 1.0},
    ]


def write_reference_artifact(
    directory: Path,
    *,
    prompt_bytes: bytes,
    chat_template_token_ids: list[int],
    expected: int = 7,
    name: str = "independent-reference.json",
    implementation_label: str = "reference-implementation",
    source_manifest_label: str = "reference-source-manifest",
    positions: list[dict[str, object]] | None = None,
) -> Path:
    spec = core.load_object(PACKAGE / "future-probe-spec.json")
    producer = copy.deepcopy(
        spec["independent_reference_requirements"]["authorized_reference_producer"]
    )
    if implementation_label != "reference-implementation":
        producer["implementation_sha256"] = digest(implementation_label)
    if source_manifest_label != "reference-source-manifest":
        producer["source_manifest_sha256"] = digest(source_manifest_label)
    if positions is None:
        alternate = 9 if expected != 9 else 7
        positions = []
        for ordinal in range(core.GENERATION_BUDGET):
            positions.append(
                {
                    "expected_argmax": expected,
                    "top_k": top_k(expected, alternate),
                    "prompt_bytes_sha256": core.sha256_bytes(prompt_bytes),
                    "chat_template_token_ids_sha256": hashlib.sha256(
                        core.canonical_bytes(chat_template_token_ids)
                    ).hexdigest(),
                    "embedding_input_ids_sha256": digest("inputs"),
                    "per_layer_state_digests": [
                        digest(f"state-{ordinal}-{layer}") for layer in range(24)
                    ],
                    "per_layer_kv_digests": [
                        digest(f"kv-{ordinal}-{layer}") for layer in range(24)
                    ],
                    "logits_sha256": digest(f"logits-{ordinal}"),
                }
            )
    artifact = {
        "schema": "ace2-r6-cf11-independent-reference-artifact-v1",
        "status": "SEALED_BEFORE_PRODUCT_EXECUTION",
        "producer": producer,
        "probe_spec_sha256": core.sha256_file(PACKAGE / "future-probe-spec.json"),
        "model_sha256": spec["bindings"]["official_model_sha256"],
        "tokenizer_json_sha256": spec["bindings"]["tokenizer_json_sha256"],
        "full_logit_count": core.MODEL_VOCABULARY_SIZE,
        "positions": positions,
    }
    path = directory / name
    path.write_bytes(core.canonical_bytes(artifact))
    path.chmod(0o444)
    return path


def write_attested_reference(
    directory: Path,
    *,
    prompt_bytes: bytes,
    chat_template_token_ids: list[int],
    expected: int = 7,
    name: str = "independent-reference.json",
    positions: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    reference_path = write_reference_artifact(
        directory,
        prompt_bytes=prompt_bytes,
        chat_template_token_ids=chat_template_token_ids,
        expected=expected,
        name=name,
        positions=positions,
    )
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    spec = core.load_object(PACKAGE / "future-probe-spec.json")
    authority = spec["independent_reference_requirements"][
        "execution_attestation_authority"
    ]
    authority.update(
        {
            "status": "BOUND_BY_FUTURE_EXECUTION_AUTHORITY",
            "key_id": "FIXTURE-ACTIVE-1",
            "verification_key_base64": base64.b64encode(public_key).decode("ascii"),
            "verification_key_sha256": hashlib.sha256(public_key).hexdigest(),
        }
    )
    artifact = core.load_object(reference_path)
    payload = {
        "reference_artifact_sha256": core.sha256_file(reference_path),
        "producer_implementation_sha256": artifact["producer"][
            "implementation_sha256"
        ],
        "producer_source_manifest_sha256": artifact["producer"][
            "source_manifest_sha256"
        ],
        "future_probe_spec_sha256": core.sha256_file(
            PACKAGE / "future-probe-spec.json"
        ),
        "model_sha256": spec["bindings"]["official_model_sha256"],
        "tokenizer_json_sha256": spec["bindings"]["tokenizer_json_sha256"],
        "process_argv_sha256": digest("reference-process-argv"),
        "process_exit_code": 0,
        "product_execution_started": False,
        "product_positions_consumed": False,
    }
    signed = {
        "schema": "ace2-r6-cf11-independent-reference-execution-attestation-v1",
        "status": "PASS_REFERENCE_PRODUCER_EXECUTED_BEFORE_PRODUCT",
        "trust_domain": "EXTERNAL_FUTURE_EXECUTION_ATTESTER",
        "signer": {
            "algorithm": "Ed25519",
            "key_id": authority["key_id"],
            "verification_key_sha256": authority["verification_key_sha256"],
            "private_key_access": (
                "INACCESSIBLE_TO_PRODUCT_AND_REFERENCE_ARTIFACT_CODE"
            ),
        },
        "payload": payload,
    }
    attestation = {
        **signed,
        "signature_base64": base64.b64encode(
            private_key.sign(core.canonical_bytes(signed))
        ).decode("ascii"),
    }
    attestation_path = directory / f"{name}.attestation.json"
    attestation_path.write_bytes(core.canonical_bytes(attestation))
    attestation_path.chmod(0o444)
    return reference_path, attestation_path, spec


@contextmanager
def use_future_spec(spec: dict[str, object]):
    load_object = core.load_object

    def load(path: Path) -> dict[str, object]:
        if path == PACKAGE / "future-probe-spec.json":
            return copy.deepcopy(spec)
        return load_object(path)

    with mock.patch.object(core, "load_object", side_effect=load):
        yield


def position(
    ordinal: int,
    reference_artifact_path: Path,
    *,
    selected: int = 7,
) -> dict[str, object]:
    reference_payload = copy.deepcopy(
        core.load_object(reference_artifact_path)["positions"][ordinal]
    )
    reference = {
        **reference_payload,
        "source_artifact_sha256": core.sha256_file(reference_artifact_path),
    }
    product_top_k = copy.deepcopy(reference_payload["top_k"])
    selected_entry = next(
        item for item in product_top_k if item["token_id"] == selected
    )
    product = {
        "prompt_bytes_sha256": reference_payload["prompt_bytes_sha256"],
        "chat_template_token_ids_sha256": reference_payload[
            "chat_template_token_ids_sha256"
        ],
        "embedding_input_ids_sha256": reference_payload[
            "embedding_input_ids_sha256"
        ],
        "per_layer_state_digests": reference_payload[
            "per_layer_state_digests"
        ],
        "per_layer_kv_digests": reference_payload["per_layer_kv_digests"],
        "logits_sha256": reference_payload["logits_sha256"],
        "top_k": product_top_k,
        "selected_rank": selected_entry["rank"],
        "selected_margin": (
            1.0 if selected_entry["rank"] == 1 else -1.0
        ),
        "source_artifact_sha256": digest("product-artifact"),
        "producer_implementation_sha256": digest("product-implementation"),
        "producer_source_manifest_sha256": digest("product-source-manifest"),
    }
    return {
        "ordinal": ordinal,
        "absolute_position": 36 + ordinal,
        "selected_token_id": selected,
        "reference": reference,
        "product": product,
        "rtl": {
            "source": "AUTHENTICATED_RTL_OR_PRODUCT_LOGIT_TILES",
            "source_artifact_sha256": digest("rtl-artifact"),
            "logits_sha256": reference_payload["logits_sha256"],
            "top_k": copy.deepcopy(product_top_k),
        },
        "cache_comparison": {
            "cache_free_logits_sha256": reference_payload["logits_sha256"],
            "cacheful_logits_sha256": reference_payload["logits_sha256"],
            "cache_free_selected_token_id": selected,
            "cacheful_selected_token_id": selected,
        },
    }


class Cf11DiagnosticTests(unittest.TestCase):
    def test_formatting_only_is_rejected_and_semantic_unicode_passes(self) -> None:
        reproduction = core.load_object(PACKAGE / "cf10-decode-reproduction.json")
        decoded = reproduction["decode"]["decoded_text"]
        self.assertEqual(
            core.classify_decoded_text(decoded),
            "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8",
        )
        self.assertFalse(core.readable_text("**\n**\n\n**\n"))
        self.assertFalse(core.readable_text("---\n\t"))
        self.assertTrue(core.readable_text("Grüße 世界 — registers"))

    def test_official_cf10_decode_reproduction_is_exact_and_nonexecuting(self) -> None:
        record = core.load_object(PACKAGE / "cf10-decode-reproduction.json")
        self.assertEqual(record["source"]["token_ids"], [1019, 1019, 198, 1019])
        self.assertEqual(record["decode"]["call_count"], 1)
        self.assertEqual(
            record["decode"]["classification"],
            "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8",
        )
        self.assertEqual(
            bytes.fromhex(record["decode"]["utf8_bytes_hex"]).decode("utf-8"),
            record["decode"]["decoded_text"],
        )
        self.assertEqual(record["execution"]["model_generation"], "NOT_EXECUTED")
        self.assertEqual(record["execution"]["rtl"], "NOT_EXECUTED")

    def test_arbitrary_utf8_prompt_bytes_are_persisted_exactly(self) -> None:
        value = "Grüße 世界 — registers".encode("utf-8")
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            reference_path, attestation_path, spec = write_attested_reference(
                Path(temporary),
                prompt_bytes=value,
                chat_template_token_ids=[1, 2, 3],
            )
            path = Path(temporary) / "evidence.json"
            with use_future_spec(spec):
                probe.FutureProbeJournal.create(
                    path,
                    prompt_bytes=value,
                    chat_template_token_ids=[1, 2, 3],
                    reference_artifact_path=reference_path,
                    reference_attestation_path=attestation_path,
                )
            artifact = core.load_object(path)
            self.assertEqual(bytes.fromhex(artifact["prompt"]["bytes_hex"]), value)
            self.assertEqual(artifact["prompt"]["utf8_text"], value.decode("utf-8"))

    def test_evidence_survives_every_failure_stage(self) -> None:
        for stage in core.FAILURE_STAGES:
            with self.subTest(stage=stage):
                with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
                    reference_path, attestation_path, spec = write_attested_reference(
                        Path(temporary),
                        prompt_bytes="语义 prompt".encode("utf-8"),
                        chat_template_token_ids=[1, 2, 3],
                    )
                    path = Path(temporary) / "evidence.json"
                    with use_future_spec(spec):
                        journal = probe.FutureProbeJournal.create(
                            path,
                            prompt_bytes="语义 prompt".encode("utf-8"),
                            chat_template_token_ids=[1, 2, 3],
                            reference_artifact_path=reference_path,
                            reference_attestation_path=attestation_path,
                        )
                        journal.record_failure(
                            stage, f"{stage}_FAILURE", "InjectedError"
                        )
                    artifact = core.load_object(path)
                    self.assertEqual(artifact["failures"][0]["stage"], stage)
                    self.assertEqual(
                        bytes.fromhex(artifact["prompt"]["bytes_hex"]),
                        "语义 prompt".encode("utf-8"),
                    )

    def test_interrupted_atomic_update_retains_prior_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            reference_path, attestation_path, spec = write_attested_reference(
                Path(temporary),
                prompt_bytes=b"prompt",
                chat_template_token_ids=[1, 2],
            )
            path = Path(temporary) / "evidence.json"
            with use_future_spec(spec):
                journal = probe.FutureProbeJournal.create(
                    path,
                    prompt_bytes=b"prompt",
                    chat_template_token_ids=[1, 2],
                    reference_artifact_path=reference_path,
                    reference_attestation_path=attestation_path,
                )
                before = path.read_bytes()
                with mock.patch.object(
                    probe.os, "replace", side_effect=OSError("injected")
                ):
                    with self.assertRaises(OSError):
                        journal.record_failure(
                            "HOST_DECODE", "INJECTED_FAILURE", "InjectedError"
                        )
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*")), [])

    def test_fake_reference_cannot_mask_systematic_disagreement(self) -> None:
        spec = core.load_object(PACKAGE / "future-probe-spec.json")
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            directory = Path(temporary)
            product_source = write_reference_artifact(
                directory,
                prompt_bytes=b"prompt",
                chat_template_token_ids=[1, 2],
                expected=9,
                name="product-source.json",
            )
            product_positions = [
                position(ordinal, product_source, selected=9)["product"]
                for ordinal in range(core.GENERATION_BUDGET)
            ]
            copied_positions = [
                {
                    field: copy.deepcopy(product[field])
                    for field in probe.REFERENCE_POSITION_FIELDS
                    if field != "expected_argmax"
                }
                for product in product_positions
            ]
            for copied in copied_positions:
                copied["expected_argmax"] = copied["top_k"][0]["token_id"]
            copied_path = write_reference_artifact(
                directory,
                prompt_bytes=b"prompt",
                chat_template_token_ids=[1, 2],
                name="product-evidence-exact-authorized-identity.json",
                positions=copied_positions,
            )
            fake = position(0, copied_path, selected=9)
            with self.assertRaisesRegex(
                core.DiagnosticError, "execution attestation is required"
            ):
                probe.validate_position(fake, spec, copied_path, None)

            same_provenance_path, same_attestation_path, bound_spec = (
                write_attested_reference(
                    directory,
                    prompt_bytes=b"prompt",
                    chat_template_token_ids=[1, 2],
                    name="same-product-provenance.json",
                )
            )
            same_provenance = position(0, same_provenance_path)
            authorized = bound_spec["independent_reference_requirements"][
                "authorized_reference_producer"
            ]
            same_provenance["product"]["producer_implementation_sha256"] = (
                authorized["implementation_sha256"]
            )
            with self.assertRaisesRegex(
                core.DiagnosticError, "provenance are not independent"
            ):
                probe.validate_position(
                    same_provenance,
                    bound_spec,
                    same_provenance_path,
                    same_attestation_path,
                )

            inconsistent_positions = copy.deepcopy(
                core.load_object(same_provenance_path)["positions"]
            )
            inconsistent_positions[0]["expected_argmax"] = 9
            inconsistent_path, inconsistent_attestation_path, inconsistent_spec = (
                write_attested_reference(
                    directory,
                    prompt_bytes=b"prompt",
                    chat_template_token_ids=[1, 2],
                    name="inconsistent-reference.json",
                    positions=inconsistent_positions,
                )
            )
            inconsistent = position(0, inconsistent_path)
            with self.assertRaisesRegex(
                core.DiagnosticError, "independently derived"
            ):
                probe.validate_position(
                    inconsistent,
                    inconsistent_spec,
                    inconsistent_path,
                    inconsistent_attestation_path,
                )

            reference_path, reference_attestation_path, reference_spec = (
                write_attested_reference(
                    directory,
                    prompt_bytes=b"prompt",
                    chat_template_token_ids=[1, 2],
                    name="valid-reference.json",
                )
            )
            disagreement = position(0, reference_path, selected=9)
            validated = probe.validate_position(
                disagreement,
                reference_spec,
                reference_path,
                reference_attestation_path,
            )
            self.assertEqual(
                validated["first_divergence"]["classification"],
                "SELECTED_TOKEN_DISAGREES_WITH_INDEPENDENT_ARGMAX",
            )

    def test_every_position_is_atomic_before_decode(self) -> None:
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temporary:
            reference_path, attestation_path, spec = write_attested_reference(
                Path(temporary),
                prompt_bytes=b"prompt",
                chat_template_token_ids=[1, 2],
            )
            path = Path(temporary) / "evidence.json"
            with use_future_spec(spec):
                journal = probe.FutureProbeJournal.create(
                    path,
                    prompt_bytes=b"prompt",
                    chat_template_token_ids=[1, 2],
                    reference_artifact_path=reference_path,
                    reference_attestation_path=attestation_path,
                )
                with self.assertRaisesRegex(core.DiagnosticError, "every generated"):
                    journal.mark_predecode_complete()
                for ordinal in range(core.GENERATION_BUDGET):
                    journal.record_position(position(ordinal, reference_path))
                    self.assertEqual(
                        len(core.load_object(path)["positions"]), ordinal + 1
                    )
                journal.mark_predecode_complete()
            artifact = core.load_object(path)
            self.assertEqual(
                artifact["status"],
                "ALL_REQUIRED_EVIDENCE_PERSISTED_BEFORE_DECODE",
            )
            self.assertEqual(
                artifact["decode"]["status"],
                "READY_FOR_ONE_OFFICIAL_WHOLE_SEQUENCE_CALL",
            )

    def test_static_contract_reproducibility_provenance_and_zero_state(self) -> None:
        contract = core.validate_static_contract()
        self.assertEqual(contract["execution_limit"], 0)
        provenance = core.load_object(PACKAGE / "rtl-provenance.json")
        self.assertEqual(provenance["rtl_source_file_count"], 19)
        report = core.load_object(PACKAGE / "reproducibility-report.json")
        self.assertEqual(
            report["status"], "PASS_TWO_BUILD_BYTE_MODE_EQUAL"
        )
        self.assertEqual(report["build_count"], 2)
        predecessors = core.load_object(PACKAGE / "predecessor-manifest.json")
        self.assertEqual(
            predecessors["status"], "PASS_CF07_CF08_CF09_CF10_BYTE_MODE_UNCHANGED"
        )
        construction = core.load_object(PACKAGE / "construction-report.json")
        self.assertEqual(
            construction["status"],
            "PASS_ZERO_CF11_ATTEMPT_CONSUMPTION_REGISTRY_RESULT",
        )
        for relative in construction["forbidden_execution_artifacts"]:
            self.assertFalse((PACKAGE / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


LAUNCH_SOURCE = """#!/bin/sh
printf '%s\\n' 'CF11 is construction-only: product, model-generation, endpoint, RTL, consumption, replay, resume, and relaunch are forbidden.' >&2
exit 2
"""


def manager_record() -> dict[str, Any]:
    return {
        "schema": "ace2-r6-external-manager-authority-v1",
        "status": "GRANTED",
        "authority_cardinality": 1,
        "semantic_fields": {
            "decision_id": CONSTRUCTION_AUTHORITY,
            "scope": "r6generationqualitydiagnosis11",
            "authorized_action": "CONSTRUCTION_AND_INDEPENDENT_REVIEW_ONLY",
            "identity": IDENTITY,
            "transaction_nonce": NONCE,
            "execution_authority_cardinality": 0,
            "execution_limit": 0,
            "product_execution": "FORBIDDEN",
            "model_generation": "FORBIDDEN",
            "endpoint_execution": "FORBIDDEN",
            "rtl_execution": "FORBIDDEN",
            "retry": "FORBIDDEN",
            "replay": "FORBIDDEN",
            "resume": "FORBIDDEN",
            "relaunch": "FORBIDDEN",
            "independent_l2_review": "REQUIRED_BEFORE_ANY_AUTHORITY",
            "stage1_state": "OPEN",
            "stage2": "FORBIDDEN",
            "stage_transition": "DISABLED",
        },
    }


def divergence_analysis(seal: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ace2-r6-cf11-generation-divergence-analysis-v1",
        "status": "EARLIEST_DIVERGENCE_UNRESOLVED_MISSING_INDEPENDENT_REFERENCE",
        "cf10_terminal_seal_sha256": sha256_file(TERMINAL_SEAL),
        "observed_trajectory": CF10_IDS,
        "first_evidence_supported_divergence": None,
        "classification": "NO_INDEPENDENT_REFERENCE_TRAJECTORY_PERSISTED",
        "boundaries": [
            {
                "stage": "PROMPT_CHAT_TEMPLATE_SERIALIZATION",
                "classification": "CF10_COMMON_CHAIN_MATCHED_DESCRIPTOR_IDS",
                "finding": (
                    "decode was reached only after accepted_runtime.prefill compared "
                    "retained chat-template IDs exactly with descriptor prompt IDs; "
                    "an independently serialized template was not persisted"
                ),
                "can_explain_observed_tokens": "UNRESOLVED_COMMON_CHAIN_ONLY",
            },
            {
                "stage": "EMBEDDING_INPUT_IDS",
                "classification": "INPUT_IDS_MATCHED_EMBEDDING_VALUES_UNCOMPARED",
                "finding": (
                    "descriptor and retained input IDs matched; embedding rows were "
                    "not compared with an independent authenticated reference"
                ),
                "can_explain_observed_tokens": "POSSIBLE_AFTER_MATCHED_IDS",
            },
            {
                "stage": "PER_LAYER_KV_STATE",
                "classification": "COMMON_MODEL_STATE_UNRESOLVED",
                "finding": (
                    "the reached decode proves cache-free/cacheful exact-logit "
                    "comparison completed, but per-layer state was not durably "
                    "published and no independent reference was available"
                ),
                "can_explain_observed_tokens": "POSSIBLE_COMMON_PATH_DEFECT",
            },
            {
                "stage": "RTL_LOGIT_TILES",
                "classification": "NOT_REACHED_IN_CF10",
                "finding": "CF10 endpoint invocation and completion were absent",
                "can_explain_observed_tokens": False,
            },
            {
                "stage": "TIED_LM_HEAD_RANK_SELECTION",
                "classification": "NO_INDEPENDENT_ARGMAX_OR_TOP_K_PERSISTED",
                "finding": (
                    "host selected full-vocabulary greedy IDs, but independent "
                    "expected argmax, top-k, selected rank, and margin were absent"
                ),
                "can_explain_observed_tokens": "POSSIBLE",
            },
            {
                "stage": "CACHE_PLUMBING",
                "classification": "CACHE_SPECIFIC_DIVERGENCE_NOT_OBSERVED",
                "finding": (
                    "cache-free/cacheful token IDs and full-logit digests had to "
                    "match before the reached decode; a common-path defect remains"
                ),
                "can_explain_observed_tokens": "COMMON_PATH_ONLY",
            },
            {
                "stage": "HOST_DECODE",
                "classification": "OFFICIAL_DECODE_REPRODUCED_FAITHFULLY",
                "finding": (
                    "one read-only official whole-sequence decode of the preserved "
                    "IDs reproduced formatting-only strict UTF-8 and remained rejected"
                ),
                "can_explain_observed_tokens": False,
            },
        ],
        "latency": {
            "durable_elapsed_seconds": seal["process"]["durable_elapsed_seconds"],
            "root_cause": "UNCLEAR_NO_PHASE_TIMING",
            "claim": (
                "1034.3 seconds is only wrapper elapsed time; no dominant phase "
                "or performance root cause is attributed"
            ),
        },
    }


def reference_producer_source_bytes() -> bytes:
    return textwrap.dedent(REFERENCE_PRODUCER_SOURCE).lstrip().encode("utf-8")


def reference_producer_manifest() -> dict[str, Any]:
    return {
        "schema": "ace2-r6-cf11-authorized-reference-producer-manifest-v1",
        "status": "PINNED_FOR_REVIEW_NOT_EXECUTED",
        "producer_kind": "INDEPENDENT_HOST_REFERENCE",
        "execution_domain": "SEPARATE_HOST_REFERENCE_PROCESS",
        "implementation": {
            "path": "independent_reference_producer.py",
            "sha256": hashlib.sha256(reference_producer_source_bytes()).hexdigest(),
        },
        "algorithm": (
            "OFFICIAL_QWEN25_CPU_CACHE_FREE_ARGMAX_WITH_CACHEFUL_KV_CAPTURE"
        ),
        "model_sha256": MODEL_SHA256,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "product_source_imports": "FORBIDDEN",
        "product_position_inputs": "FORBIDDEN",
        "activation_gate": "CURRENT_CF11_INDEPENDENT_L2_REVIEW_DONE",
        "execution_authority": "NOT_GRANTED",
    }


def future_probe_spec(seal: dict[str, Any]) -> dict[str, Any]:
    provenance = load_object(CF10 / "inputs/endpoint-build-provenance.json")
    descriptor = load_object(CF10 / "inputs/descriptor.json")
    producer_manifest = reference_producer_manifest()
    authorized_producer = {
        "kind": "INDEPENDENT_HOST_REFERENCE",
        "independent_of_product": True,
        "execution_domain": "SEPARATE_HOST_REFERENCE_PROCESS",
        "implementation_sha256": producer_manifest["implementation"]["sha256"],
        "source_manifest_sha256": hashlib.sha256(
            canonical_bytes(producer_manifest)
        ).hexdigest(),
    }
    return {
        "schema": "ace2-r6-cf11-hash-bound-future-probe-v1",
        "status": "SPECIFIED_NOT_EXECUTED",
        "identity": IDENTITY,
        "transaction_nonce": NONCE,
        "construction_execution_limit": 0,
        "reference_execution": "NOT_EXECUTED_CONSTRUCTION_MISSION",
        "expected_tokens": None,
        "invented_expected_tokens": False,
        "bindings": {
            "cf10_terminal_seal_sha256": sha256_file(TERMINAL_SEAL),
            "cf10_predecode_observability_sha256": sha256_file(
                TERMINAL_OBSERVABILITY
            ),
            "official_model_sha256": MODEL_SHA256,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_mapping_sha256": TOKENIZER_MAPPING_SHA256,
            "descriptor_sha256": sha256_file(CF10 / "inputs/descriptor.json"),
            "prompt_token_ids_sha256": descriptor["request"][
                "prompt_token_ids_sha256"
            ],
            "rtl_source_closure_sha256": provenance["rtl_source_closure"][
                "sha256"
            ],
            "rtl_source_file_count": 19,
            "public_parameter_count": 14,
            "public_port_count": 64,
            "terminal_review_handoff_sha256": sha256_file(
                TERMINAL_REVIEW / "round-0001.json"
            ),
        },
        "independent_reference_requirements": {
            "authorized_reference_producer": authorized_producer,
            "execution_attestation_authority": {
                "status": "REQUIRED_NOT_BOUND",
                "algorithm": "Ed25519",
                "key_id": None,
                "verification_key_base64": None,
                "verification_key_sha256": None,
                "binding_rule": (
                    "A_FUTURE_REVIEWED_EXECUTION_AUTHORITY_MUST_HASH_BIND_THE_"
                    "EXTERNAL_ATTESTER_KEY_BEFORE_REFERENCE_OR_PRODUCT_EXECUTION"
                ),
            },
            "producer_kind": "INDEPENDENT_HOST_REFERENCE",
            "independent_of_product": True,
            "artifact_schema": "ace2-r6-cf11-independent-reference-artifact-v1",
            "artifact_publication": (
                "SEPARATE_CANONICAL_JSON_FILE_SEALED_0444_BEFORE_PRODUCT_EXECUTION"
            ),
            "artifact_position_count": 4,
            "artifact_sha256_bound_by_every_position": True,
            "producer_implementation_sha256_required": True,
            "producer_source_manifest_sha256_required": True,
            "artifact_producer_must_exactly_match_authorized_reference_producer": True,
            "artifact_identity_is_not_proof_of_execution": True,
            "external_ed25519_execution_attestation_required": True,
            "reference_and_product_provenance_must_differ": True,
            "official_model_hash_required": True,
            "official_tokenizer_hash_required": True,
            "full_logit_count": 151936,
            "argmax_tie_break": "LOWEST_TOKEN_ID",
            "self_derived_product_copied_or_relabelled_reference": "REJECTED",
        },
        "atomic_predecode_requirements": [
            "exact_prompt_utf8_bytes_hex_and_sha256",
            "exact_chat_template_token_ids_and_sha256",
            "every_generated_ordinal_and_absolute_position",
            "full_selected_token_id",
            (
                "separately_sealed_independent_reference_artifact_with_"
                "expected_argmax_top_k_full_logits_digest_and_producer_provenance"
            ),
            (
                "rtl_product_top_k_logits_or_authenticated_full_logits_digest_"
                "with_product_producer_provenance"
            ),
            "selected_token_rank_and_margin",
            "twenty_four_per_layer_state_and_kv_digests",
            "cache_free_cacheful_full_logits_and_selected_id_comparison",
            "first_divergence_stage_and_classification",
        ],
        "decode_barrier": {
            "required_generated_positions": 4,
            "whole_sequence": True,
            "official_decode_call_limit": 1,
            "all_evidence_must_be_fsynced_before_decode": True,
            "formatting_only_dialogue": "REJECTED",
        },
        "failure_policy": {
            "atomic_replace_preserves_prior_evidence": True,
            "failure_evidence_published_at_every_boundary": True,
            "success_shaped_fallback": False,
        },
        "cf10_observed_stop_reason": seal["generation"]["stop_reason"],
    }


def rtl_provenance() -> dict[str, Any]:
    value = load_object(CF10 / "inputs/endpoint-build-provenance.json")
    closure = value["rtl_source_closure"]
    interface = value["public_interface"]
    if (
        len(closure["files"]) != 19
        or closure["sha256"]
        != "fc5a342475558d692e081feccc766a077fd7fdfa4e1e161ed50955d254b25d71"
        or interface["parameter_count"] != 14
        or interface["port_count"] != 64
    ):
        raise RuntimeError("CF10 exact RTL provenance differs")
    return {
        "schema": "ace2-r6-cf11-rtl-provenance-v1",
        "status": "PASS_EXACT_CF10_19_FILE_PROVENANCE_PRESERVED",
        "source_provenance_sha256": sha256_file(
            CF10 / "inputs/endpoint-build-provenance.json"
        ),
        "rtl_source_file_count": 19,
        "rtl_source_closure": closure,
        "public_interface": interface,
        "execution": "NOT_EXECUTED",
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def material_manifest(path: Path) -> dict[str, Any]:
    entries: dict[str, dict[str, object]] = {}
    for entry in sorted(path.rglob("*")):
        relative = entry.relative_to(path).as_posix()
        if relative == "reproducibility-report.json":
            continue
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
    return entries


def build_tree(
    target: Path,
    *,
    terminal_handoff: dict[str, Any],
    seal: dict[str, Any],
    decode: dict[str, Any],
    predecessor_record: dict[str, Any],
) -> None:
    target.mkdir()
    (target / "authority").mkdir()
    (target / "tests").mkdir()
    (target / "diagnostic_core.py").write_text(
        textwrap.dedent(CORE_SOURCE).lstrip(), encoding="utf-8"
    )
    (target / "future_probe.py").write_text(
        textwrap.dedent(PROBE_SOURCE).lstrip(), encoding="utf-8"
    )
    (target / "independent_reference_producer.py").write_bytes(
        reference_producer_source_bytes()
    )
    (target / "tests/test_cf11_diagnostic.py").write_text(
        textwrap.dedent(TEST_SOURCE).lstrip(), encoding="utf-8"
    )
    (target / "launch.sh").write_text(LAUNCH_SOURCE, encoding="ascii")
    write_json(target / "authority/manager-decision.json", manager_record())
    write_json(target / "cf10-decode-reproduction.json", decode)
    write_json(target / "divergence-analysis.json", divergence_analysis(seal))
    producer_manifest = reference_producer_manifest()
    write_json(target / "reference-producer-manifest.json", producer_manifest)
    spec = future_probe_spec(seal)
    write_json(target / "future-probe-spec.json", spec)
    write_json(target / "rtl-provenance.json", rtl_provenance())
    write_json(target / "predecessor-manifest.json", predecessor_record)
    evidence_binding = {
        "schema": "ace2-r6-cf11-cf10-sole-evidence-binding-v1",
        "status": "PASS_AUTHENTICATED_CF10_TERMINAL_BOUND",
        "terminal_review": {
            "mission_id": terminal_handoff["mission_id"],
            "producer_role": terminal_handoff["producer_role"],
            "review_status": terminal_handoff["review"]["status"],
            "latest_sha256": sha256_file(TERMINAL_REVIEW / "latest.json"),
            "handoff_sha256": sha256_file(TERMINAL_REVIEW / "round-0001.json"),
        },
        "terminal_seal_sha256": sha256_file(TERMINAL_SEAL),
        "predecode_observability_sha256": sha256_file(TERMINAL_OBSERVABILITY),
        "token_ids": CF10_IDS,
        "token_classifications": ["MAPPED_BASE_TOKEN_ID"] * 4,
        "stop_reason": "max_new_tokens",
        "official_whole_sequence_decode_calls": 1,
        "classification": "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8",
        "natural_exit_code": 2,
        "durable_elapsed_seconds": 1034.3,
    }
    write_json(target / "cf10-evidence-binding.json", evidence_binding)
    forbidden = [
        "attempt-0001",
        "authorization-consumed.json",
        "execution-registry.json",
        "endpoint-invocation.json",
        "completion",
        "completion-summary.json",
        "product-result.json",
    ]
    construction = {
        "schema": "ace2-r6-cf11-construction-report-v1",
        "status": "PASS_ZERO_CF11_ATTEMPT_CONSUMPTION_REGISTRY_RESULT",
        "identity": IDENTITY,
        "execution_limit": 0,
        "forbidden_execution_artifacts": forbidden,
        "attempt_count": 0,
        "consumption_count": 0,
        "registry_count": 0,
        "result_count": 0,
        "product_endpoint": "NOT_EXECUTED",
        "model_generation": "NOT_EXECUTED",
        "rtl": "NOT_EXECUTED",
        "independent_l2_review": "REQUIRED",
        "stage1_state": "OPEN",
        "stage2": "FORBIDDEN",
    }
    write_json(target / "construction-report.json", construction)
    contract = {
        "schema": "ace2-r6-cf11-generation-diagnostic-package-v1",
        "status": "CONSTRUCTED_REVIEW_REQUIRED_NOT_EXECUTABLE",
        "identity": IDENTITY,
        "transaction_nonce": NONCE,
        "construction_authority": CONSTRUCTION_AUTHORITY,
        "execution_authority_cardinality": 0,
        "execution_limit": 0,
        "product_execution": "FORBIDDEN",
        "model_generation": "FORBIDDEN",
        "endpoint_execution": "FORBIDDEN",
        "rtl_execution": "FORBIDDEN",
        "retry_replay_resume_relaunch": "FORBIDDEN",
        "future_probe_spec_sha256": hashlib.sha256(
            canonical_bytes(spec)
        ).hexdigest(),
        "authorized_reference_producer_implementation_sha256": (
            producer_manifest["implementation"]["sha256"]
        ),
        "authorized_reference_producer_manifest_sha256": hashlib.sha256(
            canonical_bytes(producer_manifest)
        ).hexdigest(),
        "reference_execution_attestation": (
            "REQUIRED_NOT_BOUND_NO_REFERENCE_ACCEPTED_BEFORE_FUTURE_AUTHORITY"
        ),
        "cf10_evidence_binding_sha256": hashlib.sha256(
            canonical_bytes(evidence_binding)
        ).hexdigest(),
        "cf10_decode_reproduction_sha256": hashlib.sha256(
            canonical_bytes(decode)
        ).hexdigest(),
        "exact_rtl_source_file_count": 19,
        "arbitrary_utf8_input": True,
        "atomic_publication": True,
        "tokenizer_domain_fix_preserved": True,
        "independent_l2_review": "REQUIRED_BEFORE_ANY_AUTHORITY",
        "stage1_state": "OPEN",
        "stage2": "FORBIDDEN",
        "stage_transition": "DISABLED",
    }
    write_json(target / "package-contract.json", contract)
    modes = {
        "authority": 0o555,
        "tests": 0o555,
        "diagnostic_core.py": 0o444,
        "future_probe.py": 0o444,
        "independent_reference_producer.py": 0o444,
        "tests/test_cf11_diagnostic.py": 0o444,
        "launch.sh": 0o555,
        "authority/manager-decision.json": 0o444,
        "cf10-decode-reproduction.json": 0o444,
        "divergence-analysis.json": 0o444,
        "future-probe-spec.json": 0o444,
        "reference-producer-manifest.json": 0o444,
        "rtl-provenance.json": 0o444,
        "predecessor-manifest.json": 0o444,
        "cf10-evidence-binding.json": 0o444,
        "construction-report.json": 0o444,
        "package-contract.json": 0o444,
    }
    for relative, mode in modes.items():
        (target / relative).chmod(mode)
    if any((target / relative).exists() for relative in forbidden):
        raise RuntimeError("CF11 execution state was created during construction")


def add_reproducibility_report(first: Path, second: Path) -> None:
    first_manifest = material_manifest(first)
    second_manifest = material_manifest(second)
    if first_manifest != second_manifest:
        raise RuntimeError("CF11 two-build byte/mode reproducibility differs")
    report = {
        "schema": "ace2-r6-cf11-two-build-reproducibility-v1",
        "status": "PASS_TWO_BUILD_BYTE_MODE_EQUAL",
        "build_count": 2,
        "material_entry_count": len(first_manifest),
        "material_manifest_sha256": hashlib.sha256(
            canonical_bytes(first_manifest)
        ).hexdigest(),
    }
    for target in (first, second):
        write_json(target / "reproducibility-report.json", report)
        (target / "reproducibility-report.json").chmod(0o444)
    if namespace_manifest(first)["entries"] != namespace_manifest(second)["entries"]:
        raise RuntimeError("CF11 complete two-build namespace differs")


def main() -> int:
    if PACKAGE.exists():
        raise RuntimeError(f"fresh CF11 namespace already exists: {PACKAGE}")
    required = (
        TERMINAL_SEAL,
        TERMINAL_OBSERVABILITY,
        TERMINAL_REVIEW / "latest.json",
        TERMINAL_REVIEW / "round-0001.json",
        CF10 / "inputs/descriptor.json",
        CF10 / "inputs/endpoint-build-provenance.json",
        *PREDECESSORS.values(),
    )
    for path in required:
        if not path.exists() or path.is_symlink():
            raise RuntimeError(f"immutable prerequisite is absent or linked: {path}")

    terminal_handoff, seal = validate_terminal_gate()
    before = {
        name: namespace_manifest(path) for name, path in PREDECESSORS.items()
    }
    decode = reproduce_decode_once()
    after = {
        name: namespace_manifest(path) for name, path in PREDECESSORS.items()
    }
    if after != before:
        raise RuntimeError("CF07-CF10 changed during read-only decode reproduction")
    predecessor_record = {
        "schema": "ace2-r6-cf11-predecessor-byte-mode-manifest-v1",
        "status": "PASS_CF07_CF08_CF09_CF10_BYTE_MODE_UNCHANGED",
        "before_construction": before,
        "after_construction": after,
    }

    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cf11-construction-", dir=PACKAGE.parent
    ) as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "build-a"
        second = temporary_root / "build-b"
        build_tree(
            first,
            terminal_handoff=terminal_handoff,
            seal=seal,
            decode=decode,
            predecessor_record=predecessor_record,
        )
        build_tree(
            second,
            terminal_handoff=terminal_handoff,
            seal=seal,
            decode=decode,
            predecessor_record=predecessor_record,
        )
        add_reproducibility_report(first, second)
        final_predecessors = {
            name: namespace_manifest(path) for name, path in PREDECESSORS.items()
        }
        if final_predecessors != before:
            raise RuntimeError("CF07-CF10 changed before atomic CF11 publication")
        os.replace(first, PACKAGE)
        directory = os.open(PACKAGE.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    print(PACKAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
