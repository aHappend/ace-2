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
IDENTITY = "stage1-w4a8-r6-generation-probe-terminal-accountability-cf17-0002"
PACKAGE = BUILD_ROOT / IDENTITY
STATE = BUILD_ROOT / f"{IDENTITY}-authority-state-0001"
ATTEMPT = STATE / "attempt-0001"
CF16_IDENTITY = (
    "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0002"
)
CF16_PACKAGE = BUILD_ROOT / CF16_IDENTITY
CF16_STATE = BUILD_ROOT / f"{CF16_IDENTITY}-authority-state-0001"
HOST_EVIDENCE = CF16_STATE / "attempt-0001/product/product-rtl-evidence.json"
CF15 = (
    BUILD_ROOT
    / "stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0001"
)
MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/5387edbfef70/mission.json"
)
REVIEW_ROOT = MISSION.parent
PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")
CONSTRUCTOR = Path(__file__).resolve()
RUNNER_SOURCE = ROOT / "tools/cf17_terminal_authority_runner.py"
VALIDATOR_SOURCE = ROOT / "tools/cf17_terminal_validate.py"
TEST_SOURCE = ROOT / "tools/cf17_terminal_tests.py"
EXPECTED_CF16 = {
    "authority.json": "483801efc75a7a6f19f741d2521fbdac893b8a645e629476a35fd46b458c42b8",
    "contract.json": "78c19170c068dc037296418e0944ae40d7cc5805ea8b5016dc6fbb19c8359d2d",
    "package-manifest.json": "e01b27b5d097bbe9c56cfba565abb836729093a1d98ec0695a65241ef792f921",
    "reproducibility-report.json": "b048c570a4f1569d36416b898399502a019581c97e45e5bd88c661037cfcad47",
    "review-request.json": "ebca0a37bdeea7b75f264f6a9612475c497a573a1f6fc9c545fa4f9dcfdf7f75",
    "source-constraint-manifest.json": "17ff9bef2ca97d5525ff5f92ecdee8c724c8abdf316bd35bc128e92e9c5e80e3",
}
EXPECTED_CF16_STATE = {
    "attempt-0001/execution-journal.json": "34e0ed43bdfe99dadf1d61cc4ca7d2e15c04d59c69aea7254319652f3079ff21",
    "attempt-0001/product/product-rtl-evidence.json": "e3ec582756eabce3b6381f6af49f4c10d58a3f5816c8f54ca848bbe17a104e41",
    "attempt-0001/product/rtl-completion/commands.jsonl": "223daf10fb68bbe0591fe0bbc4c62406bb5a126c207132dda31ea9ca33fc509d",
    "attempt-0001/product/rtl-completion/progress.journal": "7961abd288e2fb836f672ba4086df73cf716a05fec9ef2a2f1fdd2e42579cce3",
    "attempt-0001/product/rtl-completion/progress.json": "2ea485a05aa7d2fe5bd37feb86580d7f3457937ee11e60ff9790b4309e264176",
    "authorization-consumed.json": "9c78458e43ae33d4da4c0f337bebf934bddb4ba9c4e389547e15d08b85983e4b",
    "consumption-receipt.json": "c6c0143321072c2df2439d835ed73b5469444ba4788e48f5f0652883af745b0d",
    "execution-claim.json": "6985ac4ae55a70f9ec5a880fd34a2455ceff870d3a195c378c7981e870a26417",
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
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in read-only input: {path}")
        if path.is_file():
            records[path.relative_to(ROOT).as_posix()] = file_record(path)
    return records


def write_file(path: Path, value: bytes | str, mode: int = 0o444) -> None:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def verify_manifest_closure(package: Path) -> None:
    manifest = load_object(package / "package-manifest.json")
    expected = manifest.get("files")
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    actual = {
        path.relative_to(package).as_posix(): file_record(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.relative_to(package).as_posix() not in excluded
    }
    if actual != expected:
        raise RuntimeError(f"immutable package closure changed: {package.name}")


def authenticate_inputs() -> None:
    for relative, expected in EXPECTED_CF16.items():
        if sha256_file(CF16_PACKAGE / relative) != expected:
            raise RuntimeError(f"CF16 immutable package anchor changed: {relative}")
    verify_manifest_closure(CF16_PACKAGE)
    for relative, expected in EXPECTED_CF16_STATE.items():
        if sha256_file(CF16_STATE / relative) != expected:
            raise RuntimeError(f"CF16 immutable evidence changed: {relative}")
    actual_state = {
        path.relative_to(CF16_STATE).as_posix()
        for path in CF16_STATE.rglob("*")
        if path.is_file()
    }
    if actual_state != set(EXPECTED_CF16_STATE):
        raise RuntimeError("CF16 consumed-state exact file set changed")
    if (CF16_STATE / "terminal-seal.json").exists():
        raise RuntimeError("CF16 terminal state changed after bounded diagnosis")
    if sha256_file(CF15 / "package-manifest.json") != (
        "c2e08aaa88051446748d9038d8147719c4d55b1d624e7b35633a87392a655fb4"
    ):
        raise RuntimeError("accepted CF15 package anchor changed")
    verify_manifest_closure(CF15)
    if not MISSION.is_file():
        raise RuntimeError("CF17 immutable mission is absent")


def process_matches(fragment: str) -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if fragment.encode("utf-8") in raw:
            matches.append(int(entry.name))
    return sorted(matches)


def build_context() -> dict[str, Any]:
    bindings = load_object(CF15 / "bindings.json")
    endpoint_prefix = bindings["endpoint"]["argv_prefix"]
    endpoint_argv = list(endpoint_prefix) + [
        "--output",
        str(ATTEMPT / "endpoint-output"),
    ]
    launch_argv = [
        str(PYTHON),
        "-B",
        str(PACKAGE / "authority_runner.py"),
        "--authority",
        str(PACKAGE / "authority.json"),
    ]
    host = load_object(HOST_EVIDENCE)
    positions = host.get("host", {}).get("positions", [])
    if len(positions) != 4:
        raise RuntimeError("CF16 read-only host capture lacks four positions")
    summary = [
        {
            "ordinal": position["ordinal"],
            "absolute_position": position["absolute_position"],
            "selected_token_id": position["selected_token_id"],
            "layer_state_contract_sha256": position[
                "layer_state_contract_sha256"
            ],
        }
        for position in positions
    ]
    cf16_processes = process_matches(CF16_IDENTITY)
    if cf16_processes:
        raise RuntimeError("CF16 process remains; replay or resume is forbidden")
    if STATE.exists():
        raise RuntimeError("CF17 fresh authority state already exists")
    model_link = Path(endpoint_prefix[endpoint_prefix.index("--model") + 1])
    endpoint_paths = {
        "harness": Path(endpoint_prefix[0]),
        "runtime_package": Path(endpoint_prefix[endpoint_prefix.index("--package") + 1]),
        "memory_image": Path(endpoint_prefix[endpoint_prefix.index("--image") + 1]),
        "model": model_link.resolve(strict=True),
    }
    return {
        "bindings": bindings,
        "endpoint_argv": endpoint_argv,
        "launch_argv": launch_argv,
        "host_positions": summary,
        "cf16_package_inventory": tree_records(CF16_PACKAGE),
        "cf16_state_inventory": tree_records(CF16_STATE),
        "cf16_processes": cf16_processes,
        "endpoint_paths": endpoint_paths,
        "endpoint_path_bindings": {
            str(model_link): str(endpoint_paths["model"]),
        },
    }


def authority(context: dict[str, Any]) -> dict[str, object]:
    endpoint_argv = context["endpoint_argv"]
    launch_argv = context["launch_argv"]
    return {
        "schema": "ace2-r6-cf17-exactly-once-terminal-accountability-authority-v1",
        "identity": IDENTITY,
        "status": "GRANTED_PENDING_INDEPENDENT_REVIEWER_DONE",
        "authority_cardinality": 1,
        "execution_limit": 1,
        "authorized_action": "RUN_RTL_ENDPOINT_ONCE_AGAINST_READ_ONLY_CF16_HOST_CAPTURE",
        "consumption_evidence_namespace": str(STATE),
        "output_namespace": str(ATTEMPT),
        "launch_invocation": {
            "argv": launch_argv,
            "argv_sha256": sha256_bytes(canonical_bytes(launch_argv)),
            "local_only": True,
        },
        "rtl_endpoint_invocation": {
            "argv": endpoint_argv,
            "argv_sha256": sha256_bytes(canonical_bytes(endpoint_argv)),
            "timeout_seconds": context["bindings"]["endpoint"]["timeout_seconds"],
            "software_fallback": "FORBIDDEN",
            "resume": "FORBIDDEN",
            "path_bindings": context["endpoint_path_bindings"],
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "HOME": "/home/argustest",
            "OMP_NUM_THREADS": "1",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "activation_gate": {
            "mission_id": "5387edbfef70",
            "mission_path": str(MISSION),
            "mission_sha256": sha256_file(MISSION),
            "review_root": str(REVIEW_ROOT),
            "required_status": "INDEPENDENT_REVIEWER_DONE",
        },
        "cf16_host_capture": {
            "path": str(HOST_EVIDENCE),
            **file_record(HOST_EVIDENCE),
            "position_count": 4,
            "layer_state_count_per_position": 24,
            "positions": context["host_positions"],
            "usage": "READ_ONLY_AGREEMENT_INPUT_NO_MODEL_REEXECUTION",
        },
        "cf16_predecessor": {
            "identity": CF16_IDENTITY,
            "package_anchors": EXPECTED_CF16,
            "state_anchors": EXPECTED_CF16_STATE,
            "authority_state_mutation": "FORBIDDEN",
            "terminal_seal_at_construction": "ABSENT_BOUNDED_FAILURE_EVIDENCE",
        },
        "post_consumption_publication": {
            "endpoint_outcome": "FSYNCED_RECEIPT_OR_EXPLICIT_FSYNCED_FAILURE_CAPTURE",
            "terminal": "PRIMARY_OR_DISTINCT_FSYNCED_FALLBACK_REQUIRED",
            "every_runner_controlled_outcome": True,
        },
        "cf16_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "software_fallback": "FORBIDDEN",
        "model_reexecution": "FORBIDDEN",
        "reference_reexecution": "FORBIDDEN",
        "stage2": "FORBIDDEN",
        "u280_s7_ssh": "FORBIDDEN",
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
        "non_sram_area_cap_mm2": 2.0,
        "minimum_frequency_mhz": 100,
    }


def source_manifest(context: dict[str, Any]) -> dict[str, object]:
    sources = {}
    for path, scope in (
        (CONSTRUCTOR, "repository"),
        (RUNNER_SOURCE, "repository_template"),
        (VALIDATOR_SOURCE, "repository_template"),
        (TEST_SOURCE, "repository_template"),
        (MISSION, "external_canonical"),
        (CF15 / "bindings.json", "accepted_endpoint_binding"),
        (CF16_PACKAGE / "package-manifest.json", "immutable_cf16_package"),
        (HOST_EVIDENCE, "immutable_cf16_evidence"),
        (
            CF16_STATE / "attempt-0001/product/rtl-completion/commands.jsonl",
            "immutable_cf16_partial_endpoint_evidence",
        ),
        (
            CF16_STATE / "attempt-0001/product/rtl-completion/progress.json",
            "immutable_cf16_partial_endpoint_evidence",
        ),
    ):
        relative = path.as_posix() if path.is_absolute() and not path.is_relative_to(ROOT) else path.relative_to(ROOT).as_posix()
        sources[relative] = {"scope": scope, **file_record(path)}
    for name, path in context["endpoint_paths"].items():
        relative = path.as_posix() if not path.is_relative_to(ROOT) else path.relative_to(ROOT).as_posix()
        sources[relative] = {"scope": f"rtl_endpoint_{name}", **file_record(path)}
    return {
        "schema": "ace2-r6-cf17-source-constraint-manifest-v1",
        "files": dict(sorted(sources.items())),
        "path_bindings": context["endpoint_path_bindings"],
        "rtl_change": "NONE",
        "constraint_change": "NONE",
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
        "non_sram_area_cap_mm2": 2.0,
        "minimum_frequency_mhz": 100,
    }


def freeze_tree(package: Path) -> None:
    for path in package.rglob("*"):
        if path.is_file():
            path.chmod(
                0o555
                if path.name in {"authority_runner.py", "validate_package.py"}
                else 0o444
            )
    for directory in sorted(
        (path for path in package.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    package.chmod(0o555)


def materialize(package: Path, context: dict[str, Any]) -> None:
    package.mkdir()
    write_file(package / "authority_runner.py", RUNNER_SOURCE.read_bytes(), 0o555)
    write_file(package / "validate_package.py", VALIDATOR_SOURCE.read_bytes(), 0o555)
    write_file(
        package / "tests/test_cf17_terminal_accountability.py",
        TEST_SOURCE.read_bytes(),
    )
    write_file(
        package / "Makefile",
        (
            ".PHONY: check\n"
            "check:\n"
            f"\t{PYTHON} -B -c \"from pathlib import Path; [compile(path.read_bytes(), str(path), 'exec') for path in (Path('authority_runner.py'), Path('validate_package.py'))]\"\n"
            f"\t{PYTHON} -B validate_package.py\n"
            f"\t{PYTHON} -B -m unittest discover -s tests -v\n"
        ),
    )
    auth = authority(context)
    write_file(package / "authority.json", canonical_bytes(auth))
    write_file(
        package / "contract.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf17-terminal-accountability-contract-v1",
                "identity": IDENTITY,
                "cf16_evidence": "EXACT_HASH_BOUND_READ_ONLY",
                "cf16_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
                "host_input": "FOUR_POSITIONS_24_EXECUTED_LAYER_STATES_EACH",
                "endpoint": "RTL_HARNESS_ONLY_NO_SOFTWARE_FALLBACK",
                "endpoint_outcome": "FSYNCED_RECEIPT_OR_EXPLICIT_FAILURE_CAPTURE",
                "terminal_publication": "PRIMARY_OR_DISTINCT_FSYNCED_FALLBACK_REQUIRED",
                "authority_state_mutation": "FRESH_CF17_ONLY_CF16_FORBIDDEN",
                "stage2_u280_s7_ssh": "FORBIDDEN",
            }
        ),
    )
    write_file(
        package / "predecessor-cf16-package-inventory.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf17-cf16-package-read-only-inventory-v1",
                "identity": CF16_IDENTITY,
                "mutation": "FORBIDDEN",
                "entries": context["cf16_package_inventory"],
            }
        ),
    )
    write_file(
        package / "predecessor-cf16-state-inventory.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf17-cf16-state-read-only-inventory-v1",
                "identity": CF16_IDENTITY,
                "mutation": "FORBIDDEN",
                "entries": context["cf16_state_inventory"],
            }
        ),
    )
    write_file(
        package / "preconstruction-zero-state.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf17-preconstruction-zero-state-v1",
                "identity": IDENTITY,
                "status": "PASS_FRESH_CF17_STATE_CF16_CONSUMED_STATE_READ_ONLY",
                "cf17_state_exists": False,
                "cf16_terminal_seal_exists": False,
                "cf16_matching_processes": context["cf16_processes"],
                "cf16_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            }
        ),
    )
    source = source_manifest(context)
    write_file(package / "source-constraint-manifest.json", canonical_bytes(source))
    write_file(
        package / "source-constraint-manifest.sha256",
        f"{sha256_file(package / 'source-constraint-manifest.json')}  source-constraint-manifest.json\n",
    )
    write_file(
        package / "execution-backlog.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf17-execution-backlog-v1",
                "identity": IDENTITY,
                "status": "BLOCKED_PENDING_INDEPENDENT_REVIEWER_DONE",
                "action": "RUN_RTL_ENDPOINT_ONCE_AGAINST_READ_ONLY_CF16_HOST_CAPTURE",
                "construction_or_review_execution": "FORBIDDEN",
            }
        ),
    )
    write_file(
        package / "construction-report.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf17-construction-report-v1",
                "status": "CONSTRUCTED_NOT_EXECUTED",
                "cf16_package_and_state": "READ_ONLY_EXACT_HASH_BOUND",
                "model_tokenizer_reference_rtl_endpoint_stage2_u280_s7_ssh": "NOT_EXECUTED",
                "software_fallback": "FORBIDDEN",
            }
        ),
    )

    late = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
        "reproducibility-report.json",
    }
    for path in package.rglob("*"):
        if path.is_file() and path.name not in late:
            path.chmod(
                0o555
                if path.name in {"authority_runner.py", "validate_package.py"}
                else 0o444
            )
    core = {
        path.relative_to(package).as_posix(): file_record(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name not in late
    }
    write_file(
        package / "reproducibility-report.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf17-reproducibility-v1",
                "status": "PASS_TWO_MATERIALIZATIONS_DIRECT_FILE_SET_BYTES_MODE_SIZE",
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
                "schema": "ace2-r6-cf17-package-manifest-v1",
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
            "predecessor-cf16-package-inventory.json",
            "predecessor-cf16-state-inventory.json",
            "reproducibility-report.json",
            "source-constraint-manifest.json",
        )
    }
    write_file(
        package / "review-request.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf17-independent-review-request-v1",
                "identity": IDENTITY,
                "status": "PENDING_INDEPENDENT_REVIEWER",
                "activation_before_reviewer_done": "FORBIDDEN",
                "execution_during_review": "FORBIDDEN",
                "required_checks": [
                    "CF16_0002_PACKAGE_AND_CONSUMED_EVIDENCE_EXACT_HASH_BOUND_UNCHANGED",
                    "CF16_REPLAY_RESUME_RELAUNCH_FORBIDDEN_AND_NO_PROCESS",
                    "FOUR_POSITION_24_LAYER_HOST_CAPTURE_READ_ONLY_INPUT",
                    "FRESH_CARDINALITY_ONE_CF17_AUTHORITY",
                    "RTL_ENDPOINT_ONLY_NO_MODEL_OR_SOFTWARE_FALLBACK",
                    "FSYNCED_ENDPOINT_RECEIPT_OR_EXPLICIT_FAILURE_CAPTURE",
                    "PRIMARY_OR_DISTINCT_FALLBACK_TERMINAL_SEAL",
                    "NON_MODEL_RECEIPT_AND_TERMINAL_PUBLICATION_FAILURE_TESTS",
                    "NO_STAGE2_U280_S7_SSH_OR_CORE_RTL_CHANGE",
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
        raise RuntimeError("CF17 materialization file sets differ")
    for relative in first_paths:
        left = first / relative
        right = second / relative
        if (
            left.read_bytes() != right.read_bytes()
            or stat.S_IMODE(left.stat().st_mode)
            != stat.S_IMODE(right.stat().st_mode)
            or left.stat().st_size != right.stat().st_size
        ):
            raise RuntimeError(f"CF17 materialization differs: {relative}")


def main() -> None:
    if PACKAGE.exists():
        raise RuntimeError(f"refusing to replace existing package: {PACKAGE}")
    if STATE.exists():
        raise RuntimeError(f"refusing construction with existing state: {STATE}")
    authenticate_inputs()
    context = build_context()
    with tempfile.TemporaryDirectory(
        prefix=".cf17-build-", dir=PACKAGE.parent
    ) as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "first"
        second = temporary_root / "second"
        materialize(first, context)
        materialize(second, context)
        compare_materializations(first, second)
        first.chmod(0o755)
        first.rename(PACKAGE)
        PACKAGE.chmod(0o555)
    print(f"CONSTRUCTED_CF17_NOT_EXECUTED {PACKAGE}")


if __name__ == "__main__":
    main()
