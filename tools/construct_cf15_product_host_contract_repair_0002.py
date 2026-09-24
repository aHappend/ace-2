#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path

import construct_cf15_product_host_contract_repair as base


PREDECESSOR_PACKAGE_NAME = (
    "stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0001"
)
PACKAGE_NAME = (
    "stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0002"
)
PREDECESSOR_PACKAGE = base.BUILD_ROOT / PREDECESSOR_PACKAGE_NAME
DEFAULT_PACKAGE = base.BUILD_ROOT / PACKAGE_NAME
CONSTRUCTOR = Path(__file__).resolve()
DERIVED_FILES = (
    "reproducibility-report.json",
    "package-manifest.json",
    "package-manifest.sha256",
    "review-request.json",
    "review-request.sha256",
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_file(path: Path, value: bytes | str, mode: int = 0o444) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlink in materialization: {path}")
    if path.exists():
        path.unlink()
    base.write_file(path, value, mode)


def regular_file_names(root: Path) -> tuple[str, ...]:
    names = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in materialization: {path}")
        if path.is_file():
            names.append(path.relative_to(root).as_posix())
    return tuple(names)


def direct_compare(
    first: Path, second: Path
) -> dict[str, dict[str, object]]:
    first_names = regular_file_names(first)
    second_names = regular_file_names(second)
    if first_names != second_names:
        raise RuntimeError("CF15 materialization file sets differ")

    records: dict[str, dict[str, object]] = {}
    for relative in first_names:
        first_path = first / relative
        second_path = second / relative
        first_stat = first_path.lstat()
        second_stat = second_path.lstat()
        if not stat.S_ISREG(first_stat.st_mode) or not stat.S_ISREG(
            second_stat.st_mode
        ):
            raise RuntimeError(f"non-regular materialization entry: {relative}")
        first_mode = stat.S_IMODE(first_stat.st_mode)
        second_mode = stat.S_IMODE(second_stat.st_mode)
        if first_mode != second_mode:
            raise RuntimeError(f"CF15 materialization mode differs: {relative}")
        if first_stat.st_size != second_stat.st_size:
            raise RuntimeError(f"CF15 materialization size differs: {relative}")
        first_bytes = first_path.read_bytes()
        second_bytes = second_path.read_bytes()
        if len(first_bytes) != first_stat.st_size:
            raise RuntimeError(f"first CF15 materialization changed while read: {relative}")
        if len(second_bytes) != second_stat.st_size:
            raise RuntimeError(f"second CF15 materialization changed while read: {relative}")
        if first_bytes != second_bytes:
            raise RuntimeError(f"CF15 materialization bytes differ: {relative}")
        records[relative] = {
            "comparison": "DIRECT_READ_BYTES_EQUAL",
            "mode": first_mode,
            "sha256": base.sha256_bytes(first_bytes),
            "size": first_stat.st_size,
        }
    return records


def extend_predecessor_inventory(package: Path) -> None:
    if not PREDECESSOR_PACKAGE.is_dir():
        raise RuntimeError("CF15-0001 predecessor package is absent")
    inventory_path = package / "predecessor-read-only-inventory.json"
    inventory = load_json(inventory_path)
    entries = inventory["entries"]
    if not isinstance(entries, dict):
        raise RuntimeError("predecessor inventory entries are malformed")
    for path in sorted(PREDECESSOR_PACKAGE.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"CF15-0001 predecessor contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(base.ROOT).as_posix()
            entries[relative] = base.file_record(path)
    inventory["schema"] = (
        "ace2-r6-cf15-cf12-through-cf15-0001-read-only-inventory-v2"
    )
    inventory["successor_predecessor"] = PREDECESSOR_PACKAGE_NAME
    replace_file(inventory_path, base.canonical_bytes(inventory))


def normalize_successor(package: Path) -> None:
    extend_predecessor_inventory(package)

    contract_path = package / "contract.json"
    contract = load_json(contract_path)
    if contract.get("identity") != PACKAGE_NAME:
        raise RuntimeError("CF15 successor identity was not applied")
    contract["schema"] = "ace2-r6-cf15-product-host-contract-repair-v2"
    contract["successor_of"] = PREDECESSOR_PACKAGE_NAME
    contract["successor_scope"] = (
        "REPRODUCIBILITY_EVIDENCE_DIRECT_BYTE_MODE_SIZE_REPAIR_ONLY"
    )
    replace_file(contract_path, base.canonical_bytes(contract))

    source_path = package / "source-constraint-manifest.json"
    source_manifest = load_json(source_path)
    files = source_manifest["files"]
    if not isinstance(files, dict):
        raise RuntimeError("source/constraint manifest files are malformed")
    predecessor_constructor = Path(base.__file__).resolve()
    for constructor in (predecessor_constructor, CONSTRUCTOR):
        files[constructor.relative_to(base.ROOT).as_posix()] = {
            "scope": "repository",
            **base.file_record(constructor),
        }
    source_manifest["schema"] = "ace2-r6-cf15-source-constraint-manifest-v2"
    source_manifest["successor_of"] = PREDECESSOR_PACKAGE_NAME
    replace_file(source_path, base.canonical_bytes(source_manifest))
    replace_file(
        package / "source-constraint-manifest.sha256",
        f"{base.sha256_file(source_path)}  source-constraint-manifest.json\n",
    )

    for relative in DERIVED_FILES:
        path = package / relative
        if path.exists():
            path.unlink()


def seal_package(
    package: Path, comparison: dict[str, dict[str, object]]
) -> None:
    expected_final_count = len(comparison) + len(DERIVED_FILES)
    report = {
        "schema": "ace2-r6-cf15-reproducibility-v2",
        "identity": PACKAGE_NAME,
        "successor_of": PREDECESSOR_PACKAGE_NAME,
        "status": "PASS_TWO_INDEPENDENT_MATERIALIZATIONS_DIRECT_BYTE_MODE_SIZE_IDENTICAL",
        "comparison_method": "DIRECT_PATH_READ_BYTES_WITH_EXACT_FILE_SET_MODE_AND_SIZE",
        "digest_role": "EVIDENCE_ONLY_NOT_COMPARISON_ORACLE",
        "compared_core_files": comparison,
        "final_tree_comparison": "REQUIRED_BEFORE_ATOMIC_PUBLICATION",
        "final_tree_expected_file_count": expected_final_count,
        "official_execution": False,
    }
    base.write_file(
        package / "reproducibility-report.json", base.canonical_bytes(report)
    )

    manifest_files = {
        relative: base.file_record(package / relative)
        for relative in regular_file_names(package)
    }
    package_manifest = {
        "schema": "ace2-r6-cf15-package-manifest-v2",
        "identity": PACKAGE_NAME,
        "files": manifest_files,
    }
    manifest_path = package / "package-manifest.json"
    base.write_file(manifest_path, base.canonical_bytes(package_manifest))
    base.write_file(
        package / "package-manifest.sha256",
        f"{base.sha256_file(manifest_path)}  package-manifest.json\n",
    )

    review_request = {
        "schema": "ace2-r6-cf15-canonical-independent-l2-review-request-v2",
        "identity": PACKAGE_NAME,
        "successor_of": PREDECESSOR_PACKAGE_NAME,
        "status": "PENDING_INDEPENDENT_L2",
        "activation_before_acceptance": "FORBIDDEN",
        "execution_authority": "NONE",
        "required_checks": [
            "CF14_TERMINAL_SEAL_AND_CANONICAL_REVIEW_READ_ONLY_AUTHENTICATED",
            "CF12_THROUGH_CF15_0001_PREDECESSOR_INVENTORY_UNCHANGED",
            "RUNTIME_OMITTED_LAYER_STATE_CAPTURE_DIAGNOSIS",
            "EXPLICIT_24_EXECUTED_LAYER_STATE_ORDER_SHAPE_DTYPE_HASH_CONTRACT",
            "NEGATIVE_MISSING_DUPLICATE_REORDERED_TRUNCATED_PLACEHOLDER_REJECTION",
            "PRIVATE_SAFE_ATOMIC_FAIL_CLOSED_EVIDENCE_PERSISTENCE",
            "DIRECT_BYTE_MODE_SIZE_TWO_MATERIALIZATION_COMPARISON",
            "ZERO_EXECUTION_AUTHORITY_STAGE1_OPEN_STAGE2_FORBIDDEN",
        ],
        "targets": {
            "package_manifest": base.file_record(manifest_path),
            "predecessor_inventory": base.file_record(
                package / "predecessor-read-only-inventory.json"
            ),
            "reproducibility_report": base.file_record(
                package / "reproducibility-report.json"
            ),
            "source_constraint_manifest": base.file_record(
                package / "source-constraint-manifest.json"
            ),
        },
    }
    request_path = package / "review-request.json"
    base.write_file(request_path, base.canonical_bytes(review_request))
    base.write_file(
        package / "review-request.sha256",
        f"{base.sha256_file(request_path)}  review-request.json\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to replace existing package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    base.PACKAGE_NAME = PACKAGE_NAME
    with tempfile.TemporaryDirectory(prefix=".cf15-0002-build-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "first"
        second = temporary_root / "second"
        base.materialize(first)
        base.materialize(second)
        normalize_successor(first)
        normalize_successor(second)
        core_comparison = direct_compare(first, second)
        seal_package(first, core_comparison)
        seal_package(second, core_comparison)
        final_comparison = direct_compare(first, second)
        expected_count = load_json(first / "reproducibility-report.json")[
            "final_tree_expected_file_count"
        ]
        if len(final_comparison) != expected_count:
            raise RuntimeError("CF15 final materialization file count differs")
        first.rename(output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    print(f"BUILT_CF15_DIRECT_REPRODUCIBLY {output}")


if __name__ == "__main__":
    main()
