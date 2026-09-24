from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools import run_full_qwen_command_schedule_runtime as accepted_runtime
from tools.ace2_chat_demo import (
    PINNED_EMBEDDING_OFFSET,
    build_ace2rt2_package,
    build_full_prompt_commands,
    verify_first_attention_triplet,
    verify_first_kv_publication,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness"
NON_FROZEN_TOKEN = 89779


class VerilatedFirstKvRuntimeTest(unittest.TestCase):
    def test_first_kv_publication_matches_quantized_reference(self) -> None:
        self.assertTrue(RUNTIME.is_file(), "run `make full-qwen-runtime-build` first")
        schedule = json.loads(accepted_runtime.SCHEDULE.read_text(encoding="utf-8"))
        commands = build_full_prompt_commands(
            schedule,
            prompt_token_count=1,
            max_new_tokens=1,
        )
        embedding_offset, embedding_shape = accepted_runtime.embedding_tensor_offset(
            accepted_runtime.MODEL
        )
        self.assertEqual(embedding_offset, PINNED_EMBEDDING_OFFSET)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "runtime_package.bin"
            output = root / "rtl"
            build_ace2rt2_package(
                package,
                prompt_token_ids=[NON_FROZEN_TOKEN],
                max_new_tokens=1,
                commands=commands,
                embedding_offset=embedding_offset,
                embedding_shape=embedding_shape,
            )
            completed = subprocess.run(
                [
                    str(RUNTIME),
                    "--package",
                    str(package),
                    "--image",
                    str(accepted_runtime.IMAGE),
                    "--model",
                    str(accepted_runtime.MODEL),
                    "--output",
                    str(output),
                    "--stop-after",
                    "7",
                    "--timeout-cycles",
                    "100000000",
                    "--read-response-latency-cycles",
                    "1",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            first_summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(
                first_summary["simulator_cycles_scope"], "current_process_segment"
            )
            self.assertEqual(
                first_summary["segment_simulator_cycles"],
                first_summary["cumulative_simulator_cycles"],
            )
            resumed = subprocess.run(
                [
                    str(RUNTIME),
                    "--package",
                    str(package),
                    "--image",
                    str(accepted_runtime.IMAGE),
                    "--model",
                    str(accepted_runtime.MODEL),
                    "--output",
                    str(output),
                    "--stop-after",
                    "10",
                    "--timeout-cycles",
                    "100000000",
                    "--read-response-latency-cycles",
                    "1",
                    "--resume",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            resumed_summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(resumed_summary["resume_count"], 1)
            self.assertEqual(
                resumed_summary["prior_cumulative_simulator_cycles"],
                first_summary["cumulative_simulator_cycles"],
            )
            self.assertEqual(
                resumed_summary["cumulative_simulator_cycles"],
                first_summary["cumulative_simulator_cycles"]
                + resumed_summary["segment_simulator_cycles"],
            )
            reference = verify_first_kv_publication(
                commands[:7],
                first_prompt_token=NON_FROZEN_TOKEN,
                runtime_output=output,
            )
            self.assertEqual(reference["status"], "PASS_BIT_EXACT", reference)
            self.assertEqual(reference["bytes_compared"], 272)
            self.assertIsNone(reference["address_mismatch"])
            self.assertIsNone(reference["first_payload_mismatch_offset"])
            self.assertTrue(reference["destination_digest_matches_journal"])
            attention = verify_first_attention_triplet(
                commands[:10],
                first_prompt_token=NON_FROZEN_TOKEN,
                runtime_output=output,
            )
            self.assertEqual(attention["status"], "PASS_BIT_EXACT", attention)
            self.assertEqual(attention["commands_compared"], 3)
            self.assertEqual(attention["bytes_compared"], 96)
            self.assertIsNone(attention["first_mismatch_ordinal"])
            for check in attention["checks"]:
                self.assertTrue(check["destination_digest_matches_journal"])
                self.assertTrue(check["completion_tag_matches"])
                self.assertTrue(check["completion_error_matches"])
                self.assertTrue(check["saturation_matches"])


if __name__ == "__main__":
    unittest.main()
