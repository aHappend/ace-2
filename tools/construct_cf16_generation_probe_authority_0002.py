#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build/ace2_chat_demo"
BASE_IDENTITY = "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0001"
IDENTITY = "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0002"
BASE_PACKAGE = BUILD_ROOT / BASE_IDENTITY
PACKAGE = BUILD_ROOT / IDENTITY
BASE_STATE = BUILD_ROOT / f"{BASE_IDENTITY}-authority-state-0001"
STATE = BUILD_ROOT / f"{IDENTITY}-authority-state-0001"
ATTEMPT = STATE / "attempt-0001"
CF15_IDENTITY = (
    "stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0001"
)
CF15 = BUILD_ROOT / CF15_IDENTITY
CF15_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationproberepair15/round-0002.json"
)
CF16_REVIEW_ROOT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationprobeauthority16"
)
CF16_MISSION = CF16_REVIEW_ROOT / "mission.json"
PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")
CONSTRUCTOR = Path(__file__).resolve()
SUCCESS_STATUS = "HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE"
EVIDENCE_SCHEMA = "ace2-r6-cf13-product-rtl-evidence-v1"
ENDPOINT_SCHEMA = "ace2-r6-cf13-endpoint-invocation-receipt-v1"

EXPECTED_CF15 = {
    "contract.json": "ca54aebc53f0858fb09b851f5b3e1a34b9617897233e671d32fa821d806fd36c",
    "package-manifest.json": "c2e08aaa88051446748d9038d8147719c4d55b1d624e7b35633a87392a655fb4",
    "source-constraint-manifest.json": "f48e30bcf4a892efce37b3f15849a7f57bfabf43900f61524482d9a2546df620",
    "review-request.json": "dc0ddb527867cbf01c0bbd8e205b68678b5b0cada8b3bb286432eb5234d500d1",
}
EXPECTED_CF15_REVIEW = (
    "137957f3da7fbc113870115e6f5997e69f0fbeb26459b13f78586e3ec4256148"
)
EXPECTED_BASE = {
    "authority.json": "aaa4f4aba37efe129c4f5c7a11469a10208372e8013f73f462fddc6c9ab9a373",
    "contract.json": "63451c68d194f57e258ce4e9145d84fb1663e56d340b1b1e3529d3cc07ab41fc",
    "package-manifest.json": "3a2974c6c0263c0f83bf574ee23c27e3d69619c1d678b43e0e9eccb1c9a4bf91",
    "reproducibility-report.json": "b18a60f7f419078816ef17317c64b9d439af066c2d6ea3aad327c49b97c36a92",
    "review-request.json": "d1e190c498511eeb89666c2082d70bce9c2bcb109c55a80ec6fc445ffe555f44",
    "source-constraint-manifest.json": "6302e187029444ea0e2aecfb90138e89e26fae31b0372d78242d93ab490d7304",
}


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
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular file: {path}")
    observed = path.stat()
    return {
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": sha256_file(path),
        "size": observed.st_size,
    }


def tree_records(root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in immutable tree: {relative}")
        if path.is_file():
            records[relative] = file_record(path)
    return records


def write_file(path: Path, value: bytes | str, mode: int = 0o444) -> None:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def verify_base_package() -> None:
    for relative, expected in EXPECTED_BASE.items():
        if sha256_file(BASE_PACKAGE / relative) != expected:
            raise RuntimeError(f"CF16-0001 immutable anchor changed: {relative}")
    manifest = load_object(BASE_PACKAGE / "package-manifest.json")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict):
        raise RuntimeError("CF16-0001 package manifest is malformed")
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    actual = {
        path.relative_to(BASE_PACKAGE).as_posix(): file_record(path)
        for path in sorted(BASE_PACKAGE.rglob("*"))
        if path.is_file()
        and path.relative_to(BASE_PACKAGE).as_posix() not in excluded
    }
    if actual != expected_files:
        raise RuntimeError("CF16-0001 immutable package closure changed")
    for directory in [BASE_PACKAGE, *BASE_PACKAGE.rglob("*")]:
        if directory.is_dir() and stat.S_IMODE(directory.stat().st_mode) != 0o555:
            raise RuntimeError("CF16-0001 immutable directory mode changed")


def authenticate_sources() -> None:
    verify_base_package()
    for relative, expected in EXPECTED_CF15.items():
        if sha256_file(CF15 / relative) != expected:
            raise RuntimeError(f"accepted CF15 anchor changed: {relative}")
    if sha256_file(CF15_REVIEW) != EXPECTED_CF15_REVIEW:
        raise RuntimeError("canonical CF15 independent L2 handoff changed")
    review = load_object(CF15_REVIEW)
    if (
        review.get("producer_role") != "reviewer"
        or review.get("review", {}).get("status") != "done"
    ):
        raise RuntimeError("canonical CF15 independent L2 review is not done")
    if not CF16_MISSION.is_file():
        raise RuntimeError("CF16 immutable mission is absent")


def exact_process_matches(argv_options: tuple[tuple[str, ...], ...]) -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = tuple(
            part.decode("utf-8", errors="surrogateescape")
            for part in raw.split(b"\0")
            if part
        )
        if argv in argv_options:
            matches.append(int(entry.name))
    return sorted(matches)


def authority_values() -> tuple[dict[str, Any], list[str], list[str]]:
    authority = load_object(BASE_PACKAGE / "authority.json")
    launch_argv = [
        str(PYTHON),
        "-B",
        str(PACKAGE / "authority_runner.py"),
        "--authority",
        str(PACKAGE / "authority.json"),
    ]
    product_argv = [
        str(PYTHON),
        "-B",
        str(CF15 / "product_probe.py"),
        "--output",
        str(ATTEMPT / "product/product-rtl-evidence.json"),
        "--capture",
        str(ATTEMPT / "product/product-host-error-capture.json"),
        "--endpoint-output",
        str(ATTEMPT / "product/rtl-completion"),
    ]
    authority.update(
        {
            "identity": IDENTITY,
            "consumption_evidence_namespace": str(STATE),
            "output_namespace": str(ATTEMPT),
            "launch_invocation": {
                "argv": launch_argv,
                "argv_sha256": sha256_bytes(canonical_bytes(launch_argv)),
                "local_only": True,
            },
            "product_host_invocation": {
                "argv": product_argv,
                "argv_sha256": sha256_bytes(canonical_bytes(product_argv)),
                "working_directory": str(ROOT),
                "natural_terminal_only": True,
            },
            "terminal_sealing": {
                **authority["terminal_sealing"],
                "seal_path": str(STATE / "terminal-seal.json"),
            },
            "accepted_cf15_success_evidence": {
                "decode": "FORBIDDEN_UNTIL_FSYNCED_HOST_AND_RTL_EVIDENCE",
                "endpoint_exit_code": 0,
                "endpoint_schema": ENDPOINT_SCHEMA,
                "evidence_schema": EVIDENCE_SCHEMA,
                "host_position_count": 4,
                "ordered_executed_layer_state_count_per_position": 24,
                "rtl_position_count": 4,
                "status": SUCCESS_STATUS,
            },
            "repair_predecessor": {
                "identity": BASE_IDENTITY,
                "anchors": EXPECTED_BASE,
                "mutation": "FORBIDDEN",
            },
        }
    )
    return authority, launch_argv, product_argv


OLD_VALIDATOR_BLOCK = '''def validate_product_evidence(path: Path) -> None:
    evidence = load_object(path)
    if evidence.get("status") != "COMPLETE":
        raise AuthorityError("CF15 product/RTL evidence is not complete")
    host = evidence.get("host")
    if not isinstance(host, dict) or host.get("position_count") != 4:
        raise AuthorityError("CF15 host evidence lacks four positions")
    positions = host.get("positions")
    if not isinstance(positions, list) or len(positions) != 4:
        raise AuthorityError("CF15 host position list is malformed")
    for ordinal, position in enumerate(positions):
        if not isinstance(position, dict) or position.get("ordinal") != ordinal:
            raise AuthorityError("CF15 host positions are missing or reordered")
        layer_hashes = position.get("per_layer_state_digests")
        aggregate = position.get("layer_state_contract_sha256")
        if (
            not isinstance(layer_hashes, list)
            or len(layer_hashes) != 24
            or any(not isinstance(value, str) or len(value) != 64 for value in layer_hashes)
            or not isinstance(aggregate, str)
            or len(aggregate) != 64
        ):
            raise AuthorityError("CF15 executed 24-layer state hashes are malformed")
'''

NEW_VALIDATOR_BLOCK = '''def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_product_evidence(path: Path) -> None:
    evidence = load_object(path)
    if set(evidence) != {
        "schema", "status", "host", "endpoint", "failure_capture_sha256", "decode"
    }:
        raise AuthorityError("CF15 product/RTL success evidence fields are malformed")
    if evidence.get("schema") != "ace2-r6-cf13-product-rtl-evidence-v1":
        raise AuthorityError("CF15 product/RTL evidence schema changed")
    if evidence.get("status") != "HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE":
        raise AuthorityError("CF15 product/RTL evidence lacks the accepted success status")
    if (
        evidence.get("decode") != "FORBIDDEN_UNTIL_FSYNCED_HOST_AND_RTL_EVIDENCE"
        or evidence.get("failure_capture_sha256") is not None
    ):
        raise AuthorityError("CF15 success evidence decode or failure state is malformed")
    host = evidence.get("host")
    if not isinstance(host, dict) or set(host) != {"position_count", "positions"}:
        raise AuthorityError("CF15 host evidence structure is malformed")
    positions = host.get("positions")
    if host.get("position_count") != 4 or not isinstance(positions, list) or len(positions) != 4:
        raise AuthorityError("CF15 host evidence lacks four positions")
    for ordinal, position in enumerate(positions):
        if not isinstance(position, dict) or position.get("ordinal") != ordinal:
            raise AuthorityError("CF15 host positions are missing or reordered")
        layer_hashes = position.get("per_layer_state_digests")
        if (
            not isinstance(layer_hashes, list)
            or len(layer_hashes) != 24
            or any(not is_sha256(value) for value in layer_hashes)
            or not is_sha256(position.get("layer_state_contract_sha256"))
        ):
            raise AuthorityError("CF15 executed 24-layer state hashes are malformed")
    endpoint = evidence.get("endpoint")
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "schema", "argv_sha256", "exit_code", "stdout_sha256", "stderr_sha256", "rtl_positions"
    }:
        raise AuthorityError("CF15 endpoint receipt structure is malformed")
    if endpoint.get("schema") != "ace2-r6-cf13-endpoint-invocation-receipt-v1":
        raise AuthorityError("CF15 endpoint receipt schema changed")
    if endpoint.get("exit_code") != 0 or any(
        not is_sha256(endpoint.get(key))
        for key in ("argv_sha256", "stdout_sha256", "stderr_sha256")
    ):
        raise AuthorityError("CF15 endpoint success receipt is malformed")
    rtl_positions = endpoint.get("rtl_positions")
    if not isinstance(rtl_positions, list) or len(rtl_positions) != 4:
        raise AuthorityError("CF15 endpoint receipt lacks four RTL positions")
    for ordinal, position in enumerate(rtl_positions):
        if not isinstance(position, dict) or position.get("ordinal") != ordinal:
            raise AuthorityError("CF15 RTL positions are missing or reordered")
        kv_hashes = position.get("per_layer_kv_digests")
        if (
            not isinstance(kv_hashes, list)
            or len(kv_hashes) != 24
            or any(not is_sha256(value) for value in kv_hashes)
            or position.get("authenticated_logit_tile_count") != 4748
            or not is_sha256(position.get("authenticated_logit_tile_aggregate_sha256"))
        ):
            raise AuthorityError("CF15 RTL position receipt is malformed")
'''


TEST_HELPER = '''

def success_evidence(positions: list[dict[str, object]]) -> dict[str, object]:
    endpoint_positions = [
        {
            "ordinal": ordinal,
            "absolute_position": 31 + ordinal,
            "selected_token_id": 100 + ordinal,
            "per_layer_kv_digests": ["2" * 64 for _ in range(24)],
            "authenticated_logit_tile_count": 4748,
            "authenticated_logit_tile_aggregate_sha256": "3" * 64,
        }
        for ordinal in range(4)
    ]
    return {
        "schema": "ace2-r6-cf13-product-rtl-evidence-v1",
        "status": "HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE",
        "host": {"position_count": 4, "positions": positions},
        "endpoint": {
            "schema": "ace2-r6-cf13-endpoint-invocation-receipt-v1",
            "argv_sha256": "4" * 64,
            "exit_code": 0,
            "stdout_sha256": "5" * 64,
            "stderr_sha256": "6" * 64,
            "rtl_positions": endpoint_positions,
        },
        "failure_capture_sha256": None,
        "decode": "FORBIDDEN_UNTIL_FSYNCED_HOST_AND_RTL_EVIDENCE",
    }
'''


def transformed_runner() -> str:
    source = (BASE_PACKAGE / "authority_runner.py").read_text(encoding="utf-8")
    source = source.replace(BASE_IDENTITY, IDENTITY)
    if OLD_VALIDATOR_BLOCK not in source:
        raise RuntimeError("CF16-0001 evidence validator source changed")
    return source.replace(OLD_VALIDATOR_BLOCK, NEW_VALIDATOR_BLOCK)


def transformed_validator() -> str:
    source = (BASE_PACKAGE / "validate_package.py").read_text(encoding="utf-8")
    source = source.replace(BASE_IDENTITY, IDENTITY)
    old = '    verify_inventory("predecessor-read-only-inventory.json")\n'
    new = (
        '    verify_inventory("predecessor-read-only-inventory.json")\n'
        '    verify_inventory("predecessor-cf16-read-only-inventory.json")\n'
    )
    if old not in source:
        raise RuntimeError("CF16-0001 inventory validation source changed")
    source = source.replace(old, new, 1)
    old_contract = '''        or authority.get("endpoint_optimization_candidate", {}).get("integration") != "EXCLUDED"
    ):
        raise ValidationError("CF16 exactly-once authority policy changed")
'''
    new_contract = '''        or authority.get("endpoint_optimization_candidate", {}).get("integration") != "EXCLUDED"
        or authority.get("accepted_cf15_success_evidence", {}).get("status")
        != "HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE"
        or authority.get("accepted_cf15_success_evidence", {}).get("evidence_schema")
        != "ace2-r6-cf13-product-rtl-evidence-v1"
        or authority.get("accepted_cf15_success_evidence", {}).get("endpoint_schema")
        != "ace2-r6-cf13-endpoint-invocation-receipt-v1"
    ):
        raise ValidationError("CF16 exactly-once authority policy changed")
'''
    if old_contract not in source:
        raise RuntimeError("CF16-0001 contract validator source changed")
    return source.replace(old_contract, new_contract, 1)


def transformed_tests() -> str:
    source = (BASE_PACKAGE / "tests/test_cf16_authority.py").read_text(
        encoding="utf-8"
    )
    marker = "    return authority\n\n\nclass PackageTests"
    if marker not in source:
        raise RuntimeError("CF16-0001 test helper insertion point changed")
    source = source.replace(
        marker, "    return authority\n" + TEST_HELPER + "\n\nclass PackageTests", 1
    )
    source = source.replace(
        "test_consumption_is_durable_before_process_and_natural_terminal_seals",
        "test_cf15_success_status_integration_shape_seals_completed",
        1,
    )
    old_fixture = (
        '{"status": "COMPLETE", "host": {"position_count": 4, "positions": positions}}'
    )
    if old_fixture not in source:
        raise RuntimeError("CF16-0001 success fixture source changed")
    source = source.replace(old_fixture, "success_evidence(positions)", 1)
    insertion = '''    def test_post_consumption_runner_failure_is_terminally_sealed(self) -> None:
'''
    rejection = '''    def test_legacy_complete_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            path.write_text('{"status":"COMPLETE"}\\n', encoding="utf-8")
            with self.assertRaisesRegex(
                runner.AuthorityError, "success evidence fields"
            ):
                runner.validate_product_evidence(path)

'''
    if insertion not in source:
        raise RuntimeError("CF16-0001 rejection-test insertion point changed")
    return source.replace(insertion, rejection + insertion, 1)


def zero_state_report(
    launch_argv: list[str], product_argv: list[str]
) -> dict[str, object]:
    forbidden = {
        "state_namespace": STATE,
        "registry": PACKAGE / "execution-registry.json",
        "claim": PACKAGE / "execution-claim.json",
        "attempt": PACKAGE / "attempt-0001",
        "output": PACKAGE / "output",
        "terminal": PACKAGE / "terminal-seal.json",
        "consumption_receipt": PACKAGE / "consumption-receipt.json",
        "authorization_consumed": PACKAGE / "authorization-consumed.json",
    }
    observed = {name: os.path.lexists(path) for name, path in forbidden.items()}
    process_matches = exact_process_matches((tuple(launch_argv), tuple(product_argv)))
    if any(observed.values()) or process_matches:
        raise RuntimeError("CF16-0002 namespace is not virgin")
    if STATE.resolve() == BASE_STATE.resolve():
        raise RuntimeError("CF16-0002 state namespace is not path-distinct")
    return {
        "schema": "ace2-r6-cf16-preconstruction-zero-state-v2",
        "identity": IDENTITY,
        "state_namespace": str(STATE),
        "path_distinct_from_cf16_0001_and_predecessor_states": True,
        "observed_paths_exist": observed,
        "exact_launch_or_product_process_matches": process_matches,
        "status": "PASS_ZERO_REGISTRY_CLAIM_ATTEMPT_OUTPUT_TERMINAL_RECEIPT_PROCESS_STATE",
    }


def source_manifest() -> dict[str, object]:
    sources: dict[str, dict[str, object]] = {
        CONSTRUCTOR.relative_to(ROOT).as_posix(): {
            "scope": "repository",
            **file_record(CONSTRUCTOR),
        },
        CF15_REVIEW.as_posix(): {
            "scope": "external_canonical",
            **file_record(CF15_REVIEW),
        },
        CF16_MISSION.as_posix(): {
            "scope": "external_canonical",
            **file_record(CF16_MISSION),
        },
    }
    for relative in EXPECTED_CF15:
        path = CF15 / relative
        sources[path.relative_to(ROOT).as_posix()] = {
            "scope": "repository",
            **file_record(path),
        }
    for relative in EXPECTED_BASE:
        path = BASE_PACKAGE / relative
        sources[path.relative_to(ROOT).as_posix()] = {
            "scope": "immutable_repair_predecessor",
            **file_record(path),
        }
    return {
        "schema": "ace2-r6-cf16-source-constraint-manifest-v2",
        "files": dict(sorted(sources.items())),
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "non_sram_area_cap_mm2": 2.0,
        "minimum_frequency_mhz": 100,
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
        "rtl_change": "NONE",
        "constraint_change": "NONE",
    }


def unlock_tree(package: Path) -> None:
    package.chmod(0o755)
    for path in package.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def freeze_tree(package: Path) -> None:
    for path in package.rglob("*"):
        if path.is_file():
            path.chmod(0o555 if path.name in {"authority_runner.py", "validate_package.py"} else 0o444)
    for directory in sorted(
        (path for path in package.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    package.chmod(0o555)


def materialize(package: Path) -> None:
    shutil.copytree(BASE_PACKAGE, package)
    unlock_tree(package)
    authority, launch_argv, product_argv = authority_values()
    contract = load_object(BASE_PACKAGE / "contract.json")
    contract.update(
        {
            "identity": IDENTITY,
            "future_invocation_sha256": authority["launch_invocation"]["argv_sha256"],
            "product_invocation_sha256": authority["product_host_invocation"]["argv_sha256"],
            "accepted_cf15_success_evidence": authority[
                "accepted_cf15_success_evidence"
            ],
            "repair_predecessor": authority["repair_predecessor"],
        }
    )
    write_file(package / "contract.json", canonical_bytes(contract))
    write_file(package / "authority.json", canonical_bytes(authority))
    write_file(package / "authority_runner.py", transformed_runner(), 0o555)
    write_file(package / "validate_package.py", transformed_validator(), 0o555)
    write_file(package / "tests/test_cf16_authority.py", transformed_tests())
    write_file(
        package / "preconstruction-zero-state.json",
        canonical_bytes(zero_state_report(launch_argv, product_argv)),
    )
    write_file(
        package / "predecessor-cf16-read-only-inventory.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-additive-repair-predecessor-inventory-v1",
                "identity": BASE_IDENTITY,
                "mutation": "FORBIDDEN",
                "entries": tree_records(BASE_PACKAGE),
            }
        ),
    )
    source = source_manifest()
    write_file(package / "source-constraint-manifest.json", canonical_bytes(source))
    write_file(
        package / "source-constraint-manifest.sha256",
        f"{sha256_file(package / 'source-constraint-manifest.json')}  source-constraint-manifest.json\n",
    )
    write_file(
        package / "execution-backlog.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-execution-backlog-v2",
                "identity": IDENTITY,
                "action": "EXECUTE_ACCEPTED_CF15_LOCAL_STAGE1_DIAGNOSTIC_EXACTLY_ONCE",
                "status": "BLOCKED_PENDING_CANONICAL_INDEPENDENT_L2_ACCEPTANCE",
                "separate_from_construction_and_review": True,
            }
        ),
    )
    write_file(
        package / "construction-report.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-construction-report-v2",
                "status": "CONSTRUCTED_NOT_EXECUTED",
                "model_tokenizer_product_host_rtl_endpoint_simulator_remote_gpu_stage2": "NOT_EXECUTED",
                "deterministic_non_model_checks": "REQUIRED_AFTER_PUBLICATION",
                "cf12_through_cf16_0001_and_endpoint_candidate": "READ_ONLY",
            }
        ),
    )
    write_file(
        package / "construction-repair-record.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-construction-repair-v2",
                "failure_taxonomy": "SUCCESS_ORACLE_CONTRACT_MISMATCH",
                "root_cause": (
                    "CF16_0001_REQUIRED_INVENTED_COMPLETE_STATUS_INSTEAD_OF_ACCEPTED_CF15_TERMINAL_STATUS"
                ),
                "regression": (
                    "INTEGRATION_SHAPED_NON_MODEL_EVIDENCE_WITH_HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE"
                ),
                "scope": "ADDITIVE_CF16_0002_ONLY_CF16_0001_UNCHANGED",
            }
        ),
    )

    late_files = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
        "reproducibility-report.json",
    }
    for path in package.rglob("*"):
        if path.is_file() and path.name not in late_files:
            path.chmod(
                0o555
                if path.name in {"authority_runner.py", "validate_package.py"}
                else 0o444
            )
    core = {
        path.relative_to(package).as_posix(): file_record(path)
        for path in sorted(package.rglob("*"))
        if path.is_file()
        and path.name not in late_files
    }
    write_file(
        package / "reproducibility-report.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-reproducibility-v2",
                "status": "PASS_TWO_INDEPENDENT_MATERIALIZATIONS_DIRECT_BYTES_MODE_SIZE_IDENTICAL",
                "comparison": "DIRECT_FILE_SET_BYTES_MODE_SIZE",
                "core_files": core,
                "official_execution": False,
            }
        ),
    )
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    files = {
        path.relative_to(package).as_posix(): file_record(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.relative_to(package).as_posix() not in excluded
    }
    write_file(
        package / "package-manifest.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-package-manifest-v1",
                "identity": IDENTITY,
                "files": files,
            }
        ),
    )
    write_file(
        package / "package-manifest.sha256",
        f"{sha256_file(package / 'package-manifest.json')}  package-manifest.json\n",
    )
    targets = {
        relative: file_record(package / relative)
        for relative in (
            "authority.json",
            "contract.json",
            "package-manifest.json",
            "preconstruction-zero-state.json",
            "predecessor-cf16-read-only-inventory.json",
            "reproducibility-report.json",
            "source-constraint-manifest.json",
        )
    }
    write_file(
        package / "review-request.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-canonical-independent-l2-review-request-v2",
                "identity": IDENTITY,
                "status": "PENDING_INDEPENDENT_L2",
                "activation_before_acceptance": "FORBIDDEN",
                "execution_during_review": "FORBIDDEN",
                "requested_level": "CANONICAL_INDEPENDENT_L2",
                "repair_predecessor": BASE_IDENTITY,
                "required_checks": [
                    "CF16_0001_EXACT_CLOSURE_UNCHANGED",
                    "CF15_EXACT_ANCHORS_AND_CANONICAL_REVIEW_AUTHENTICATED",
                    "VIRGIN_PATH_DISTINCT_AUTHORITY_STATE_AND_ZERO_PROCESS",
                    "EXACT_CF15_HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE_SUCCESS_STATUS",
                    "STRUCTURAL_HOST_24_LAYER_AND_ENDPOINT_RECEIPT_VALIDATION",
                    "INTEGRATION_SHAPED_NON_MODEL_SUCCESS_AND_LEGACY_STATUS_REJECTION_TESTS",
                    "ONE_NATURAL_TERMINAL_NO_RETRY_REPLAY_RESUME_RELAUNCH",
                    "ENDPOINT_PERFORMANCE_CANDIDATE_EXCLUDED",
                ],
                "targets": targets,
            }
        ),
    )
    write_file(
        package / "review-request.sha256",
        f"{sha256_file(package / 'review-request.json')}  review-request.json\n",
    )
    freeze_tree(package)


def compare_materializations(first: Path, second: Path) -> None:
    first_paths = [
        path.relative_to(first).as_posix()
        for path in sorted(first.rglob("*"))
        if path.is_file()
    ]
    second_paths = [
        path.relative_to(second).as_posix()
        for path in sorted(second.rglob("*"))
        if path.is_file()
    ]
    if first_paths != second_paths:
        raise RuntimeError("CF16-0002 materialization file sets differ")
    for relative in first_paths:
        first_path = first / relative
        second_path = second / relative
        if (
            first_path.read_bytes() != second_path.read_bytes()
            or stat.S_IMODE(first_path.stat().st_mode)
            != stat.S_IMODE(second_path.stat().st_mode)
            or first_path.stat().st_size != second_path.stat().st_size
        ):
            raise RuntimeError(f"CF16-0002 materialization differs: {relative}")


def main() -> None:
    if PACKAGE.exists():
        raise RuntimeError(f"refusing to replace existing package: {PACKAGE}")
    if STATE.exists():
        raise RuntimeError(f"refusing construction with existing state: {STATE}")
    authenticate_sources()
    with tempfile.TemporaryDirectory(prefix=".cf16-0002-build-", dir=PACKAGE.parent) as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "first"
        second = temporary_root / "second"
        materialize(first)
        materialize(second)
        compare_materializations(first, second)
        first.chmod(0o755)
        first.rename(PACKAGE)
        PACKAGE.chmod(0o555)
    print(f"CONSTRUCTED_CF16_0002_NOT_EXECUTED {PACKAGE}")


if __name__ == "__main__":
    main()
