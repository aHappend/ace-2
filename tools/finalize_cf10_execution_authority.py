#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf10"
sys.path.insert(0, str(PACKAGE))

import package_core as core
import prepare_package as prepare


METADATA = (
    "package-checks.json",
    "reproducibility-report.json",
    "authorization.json",
)


def regular_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


def inventory(
    root: Path, *, exclude: set[str] | None = None
) -> dict[str, dict[str, int | str]]:
    omitted = exclude or set()
    records: dict[str, dict[str, int | str]] = {}
    for path in regular_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in omitted:
            continue
        records[relative] = {
            "mode": stat.S_IMODE(path.stat().st_mode),
            "size": path.stat().st_size,
            "sha256": core.sha256_file(path),
        }
    return records


def write_canonical(path: Path, value: object, mode: int = 0o444) -> None:
    temporary = path.with_name(f".{path.name}.authority-tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(core.canonical_bytes(value))
    temporary.chmod(mode)
    os.replace(temporary, path)


def verify_reviewed_pre_authority() -> tuple[
    dict[str, object], dict[str, object]
]:
    core.validate_cf10_review_acceptance()
    for name in (
        "authorization.json",
        "package-checks.json",
        "reproducibility-report.json",
        "authority/manager-decision.json",
        "predecessor-manifest.json",
    ):
        core.require_file(
            PACKAGE / name,
            core.PRE_AUTHORITY_HASHES[name],
            f"reviewed pre-authority {name}",
        )
    core.validate_construction_only_record()
    core.assert_fresh_execution_state()
    return (
        core.load_object(PACKAGE / "authorization.json"),
        core.load_object(PACKAGE / "package-checks.json"),
    )


def materialize_authority() -> None:
    raw = core.canonical_bytes(core.manager_decision_record())
    if core.AUTHORITY_RECORD.exists():
        if core.AUTHORITY_RECORD.read_bytes() != raw:
            raise core.ProductError("existing CF10 execution authority differs")
    else:
        core.AUTHORITY_RECORD.write_bytes(raw)
    core.AUTHORITY_RECORD.chmod(0o444)
    core.validate_manager_authority()


def normalize_source_modes() -> None:
    for path in (
        PACKAGE / "package_core.py",
        PACKAGE / "product_driver.py",
        PACKAGE / "tests/test_product_package.py",
    ):
        path.chmod(0o444)
    for path in (PACKAGE / "prepare_package.py", PACKAGE / "launch.sh"):
        path.chmod(0o555)


def copy_package(destination: Path) -> None:
    shutil.copytree(
        PACKAGE,
        destination,
        copy_function=os.link,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".cf10-authority-*"),
    )
    for name in METADATA:
        (destination / name).unlink()


def build(
    destination: Path,
    base_authorization: dict[str, object],
    base_checks: dict[str, object],
    gates: dict[str, object],
) -> None:
    copy_package(destination)
    checks = prepare.authorized_package_checks(base_checks, gates)
    write_canonical(destination / "package-checks.json", checks)
    report_inventory = inventory(
        destination,
        exclude={"authorization.json", "reproducibility-report.json"},
    )
    compared = sorted(
        {*report_inventory, "authorization.json", "reproducibility-report.json"}
    )
    report = prepare.authorized_reproducibility_report(report_inventory, compared)
    write_canonical(destination / "reproducibility-report.json", report)
    authorization = prepare.authorized_sidecar(
        base_authorization,
        gates,
        package_checks_sha256=core.sha256_file(destination / "package-checks.json"),
        reproducibility_report_sha256=core.sha256_file(
            destination / "reproducibility-report.json"
        ),
    )
    write_canonical(destination / "authorization.json", authorization)


def main() -> int:
    base_authorization, base_checks = verify_reviewed_pre_authority()
    materialize_authority()
    normalize_source_modes()
    gates = core.validate_prerequisites()
    with tempfile.TemporaryDirectory(
        prefix=".cf10-authority-build-a-", dir=PACKAGE.parent
    ) as first:
        with tempfile.TemporaryDirectory(
            prefix=".cf10-authority-build-b-", dir=PACKAGE.parent
        ) as second:
            first_path = Path(first) / PACKAGE.name
            second_path = Path(second) / PACKAGE.name
            build(first_path, base_authorization, base_checks, gates)
            build(second_path, base_authorization, base_checks, gates)
            if inventory(first_path) != inventory(second_path):
                raise core.ProductError(
                    "CF10 authority builds differ by byte or mode"
                )
            for name in METADATA:
                source = first_path / name
                write_canonical(PACKAGE / name, core.load_object(source))
    for path in (PACKAGE / "__pycache__", PACKAGE / "tests/__pycache__"):
        if path.exists():
            shutil.rmtree(path)
    core.validate_prepared_authorization()
    core.assert_fresh_execution_state()
    print("PASS_CF10_AUTHORITY_TWO_BUILD_BYTE_MODE_EQUAL_NON_EXECUTING")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (core.ProductError, OSError, ValueError) as error:
        print(f"CF10 authority construction refused: {error}", file=sys.stderr)
        raise SystemExit(2)
