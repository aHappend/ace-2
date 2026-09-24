#!/usr/bin/env python3
"""Run the frozen V15 constructor, quality campaign, and verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-alias-safe-v10-transformer-v12-lm-head64-product-chat-w4-grouped-dynamic-scale32-w4a8-v15"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "run-all", "verify-result"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import v15_v10_transformer_v12_lm_head64_backend as backend

    if args.command == "self-test":
        result = backend.combined_self_test()
        if args.output is not None:
            output = args.output.resolve()
            backend.require(output.is_relative_to(ROOT), "V15 self-test output must remain inside the repository")
            backend.require(not output.is_relative_to(OFFICIAL_ROOT), "V15 self-test may not enter the official namespace")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(output)
        print("ACE2_OPTION_B_V10_TRANSFORMER_V12_LM_HEAD64_V15_SELF_TEST_PASS " + json.dumps(result, sort_keys=True))
        return 0
    if args.command == "run-all":
        return backend.run_all()
    result = backend.verify_result()
    print("ACE2_OPTION_B_V10_TRANSFORMER_V12_LM_HEAD64_V15_RESULT_VERIFIED " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
