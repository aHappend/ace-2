#!/usr/bin/env python3
"""Generate or verify model-derived layer-16 group-4 Scale32 RTL vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

from ace2_layer16_group4_scale32_reference import (
    ROOT,
    verify_model_vectors,
    write_model_vectors,
)


DEFAULT_OUTPUT = ROOT / "verification/generated/ace2_layer16_group4_scale32"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError("vector output must be repository-relative")
    if args.check:
        manifest = verify_model_vectors(output)
        print(
            "ACE2_LAYER16_GROUP4_SCALE32_VECTOR_CHECK_PASS "
            f"q_saturations={manifest['expected']['q_saturation_count']}"
        )
    else:
        manifest = write_model_vectors(output)
        print(
            "ACE2_LAYER16_GROUP4_SCALE32_VECTOR_GENERATION_PASS "
            f"groups={manifest['shape']['source_groups']} "
            f"q_outputs={manifest['shape']['q_outputs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

