#!/usr/bin/env python3
"""Generate or verify focused vectors for layer-16 down-projection output grouping."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "evidence/candidates/w4a8-layer16-down-proj-output-residual-source-repair-v1/"
    "candidate-0001"
)
DEFAULT_OUTPUT = ROOT / "verification/generated/ace2_layer16_down_proj_output_grouped_scale32"
HIDDEN = 896


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read(group: Path, name: str, dtype: str, count: int) -> np.ndarray:
    values = np.fromfile(group / f"{name}.bin", dtype=dtype)
    if values.size != count:
        raise ValueError(f"{name} count differs: {values.size} != {count}")
    return values


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << width) - 1
    digits = (width + 3) // 4
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def write_vectors(output: Path) -> dict[str, object]:
    result = json.loads((CANDIDATE / "result.json").read_text(encoding="utf-8"))
    group_size = int(result["rtl_validation_down_output_group_size"])
    group = CANDIDATE / f"groups/group-{group_size:03d}"
    group_count = HIDDEN // group_size
    source = _read(group, "down_source_s8", "i1", HIDDEN)
    base_scale = _read(group, "down_base_scale32", "<i8", 1)
    group_scale = _read(group, "down_group_scale32", "<i8", group_count)
    expected = _read(group, "down_group_s8", "i1", HIDDEN)
    saturation = _read(group, "down_saturation", "u1", HIDDEN)

    output.mkdir(parents=True, exist_ok=False)
    _write_hex(output / "down_source_s8.hex", source, 8)
    _write_hex(output / "down_base_scale32.hex", base_scale, 32)
    _write_hex(output / "down_group_scale32.hex", group_scale, 32)
    _write_hex(output / "expected_down_s8.hex", expected, 8)
    _write_hex(output / "expected_saturation.hex", saturation, 8)
    (output / "ace2_layer16_down_proj_output_grouped_scale32_constants.svh").write_text(
        "\n".join(
            [
                f"localparam integer MODEL_HIDDEN = {HIDDEN};",
                f"localparam integer MODEL_GROUP_SIZE = {group_size};",
                f"localparam integer MODEL_GROUP_COUNT = {group_count};",
                f"localparam integer MODEL_SATURATION_COUNT = {int(saturation.sum())};",
                "",
            ]
        ),
        encoding="ascii",
    )
    selected = result["selected_down_output_group_size"] is not None
    manifest = {
        "schema_version": 1,
        "classification": (
            "candidate_only_selected_focused_rtl_vectors"
            if selected
            else "candidate_only_best_null_focused_rtl_vectors"
        ),
        "boundary": "model.layers.16.mlp.down_proj.output_to_model.layers.16.output",
        "selected_as_repair": selected,
        "validation_candidate": {
            "down_output_group_size": group_size,
            "reference_rank": (
                result["selected_reference_rank"]
                if selected
                else result["best_observed_reference_rank"]
            ),
            "baseline_reference_rank": result["baseline_reference_rank"],
            "top_token_id": (
                result["selected_top_token_id"]
                if selected
                else result["best_observed_top_token_id"]
            ),
            "first_token_restored": result["first_token_restored"],
            "selection_reason": result["selection_reason"],
        },
        "shape": {"hidden": HIDDEN, "group_size": group_size, "groups": group_count},
        "expected": {
            "saturation_count": int(saturation.sum()),
            "down_s8_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        },
        "bindings": {
            "candidate_result": _artifact(CANDIDATE / "result.json"),
            "candidate_sha256s": _artifact(CANDIDATE / "SHA256SUMS"),
            "selected_group_result": _artifact(group / "result.json"),
            "search_runner": _artifact(
                ROOT
                / "tools/ace2_checkpoint176_layer16_down_proj_output_grouped_scale32_search.py"
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in members),
        encoding="ascii",
    )
    return manifest


def verify_vectors(output: Path) -> dict[str, object]:
    temporary = output.parent / f".{output.name}.check"
    if temporary.exists():
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
    expected = write_vectors(temporary)
    try:
        expected_files = sorted(path.name for path in temporary.iterdir())
        observed_files = sorted(path.name for path in output.iterdir())
        if expected_files != observed_files:
            raise RuntimeError("generated vector member list differs")
        for name in expected_files:
            if (temporary / name).read_bytes() != (output / name).read_bytes():
                raise RuntimeError(f"generated vector differs: {name}")
    finally:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError("vector output must be repository-relative")
    manifest = verify_vectors(output) if args.check else write_vectors(output)
    print(
        "ACE2_LAYER16_DOWN_PROJ_OUTPUT_GROUPED_SCALE32_VECTOR_"
        + ("CHECK" if args.check else "GENERATION")
        + "_PASS "
        + f"groups={manifest['shape']['groups']} "
        + f"rank={manifest['validation_candidate']['reference_rank']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
