from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import profile_ace2_chat_token_scaling as profiler


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(profiler.canonical_bytes(value))


def seal_attempt(path: Path) -> str:
    members = sorted(
        member
        for member in path.rglob("*")
        if member.is_file() and member.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    )
    raw = "".join(
        f"{profiler.sha256_file(member)}  {member.relative_to(path).as_posix()}\n"
        for member in members
    ).encode("ascii")
    (path / "SHA256SUMS").write_bytes(raw)
    root = hashlib.sha256(raw).hexdigest()
    (path / "TREE_ROOT.sha256").write_text(
        f"{root}  SHA256SUMS\n",
        encoding="ascii",
    )
    return root


def seal_oracle(path: Path) -> None:
    members = sorted(
        member for member in path.iterdir() if member.is_file() and member.name != "SHA256SUMS"
    )
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{profiler.sha256_file(member)}  {member.name}\n"
            for member in members
        ),
        encoding="ascii",
    )


class Ace2ChatTokenScalingProfilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_binding = hashlib.sha256(b"common-source").hexdigest()
        self.cases = [self.make_case(tokens) for tokens in (4, 6, 8)]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_case(self, tokens: int) -> tuple[int, Path, Path]:
        attempt = self.root / f"attempt-{tokens}"
        oracle = self.root / f"oracle-{tokens}"
        attempt.mkdir()
        oracle.mkdir()
        generated = list(range(100, 100 + tokens))
        source = self.source_binding
        latency = {
            "compile_wall_seconds": float(tokens * 2),
            "model_wall_seconds": float(tokens * 3),
            "simulation_wall_seconds": float(tokens * 5),
            "total_wall_seconds": float(tokens * 10),
        }
        write_json(
            attempt / "attempt-manifest.json",
            {
                "tokenization": {
                    "generation_bounds": {"max_new_tokens": tokens}
                },
                "source_and_rtl": {"sha256": source},
            },
        )
        write_json(
            attempt / "attempt-result.json",
            {
                "status": "PASS",
                "timed_out": False,
                "generated_token_ids": generated,
            },
        )
        write_json(
            attempt / "runtime-output/product_result.json",
            {
                "status": profiler.PRODUCT_STATUS,
                "generated_token_ids": generated,
                "software_transformer_or_logits_fallback": False,
                "process_tree_rss": {
                    "complete": True,
                    "peak_rss_kib": tokens * 1024,
                },
            },
        )
        write_json(
            attempt / "runtime-output/run_summary.json",
            {
                "generated_token_ids": generated,
                "software_transformer_or_logits_fallback": False,
            },
        )
        write_json(
            attempt / "timing.json",
            {
                "supervisor_total_wall_seconds": float(tokens * 11),
                "backend_latency": latency,
            },
        )
        tree_root = seal_attempt(attempt)
        write_json(
            oracle / "result.json",
            {
                "status": "PASS",
                "numerical_status": "PASS",
                "constraints": {"max_new_tokens": tokens},
                "attempt": {"tree_root_sha256": tree_root},
                "source_binding": {"sha256": source},
                "agreement": {
                    "generated_token_ids": generated,
                    "full_vocabulary_head_steps_checked": tokens,
                    "integer_byte_mismatches": 0,
                    "selected_token_mismatches": 0,
                    "kv_append_bytes_compared": tokens,
                    "full_cache_bytes_compared": tokens,
                    "quantized_layer_output_bytes_compared": tokens,
                    "full_vocabulary_logit_bytes_compared": tokens,
                    "retained_payload_bytes_compared": tokens,
                },
                "timing": {
                    "reused_rtl_attempt": latency,
                    "independent_oracle_host": {
                        "total_host_wall_seconds": float(tokens * 7),
                        "peak_rss_kib": tokens * 512,
                    },
                },
            },
        )
        seal_oracle(oracle)
        return tokens, attempt, oracle

    def reseal_case(self, case_index: int) -> None:
        _tokens, attempt, oracle = self.cases[case_index]
        tree_root = seal_attempt(attempt)
        result = json.loads((oracle / "result.json").read_text())
        if result["attempt"].get("tree_root_sha256") != "mixed":
            result["attempt"]["tree_root_sha256"] = tree_root
            write_json(oracle / "result.json", result)
        seal_oracle(oracle)

    def test_accepts_complete_authenticated_4_6_8_evidence(self) -> None:
        profile = profiler.build_profile(self.cases)
        self.assertEqual("PASS", profile["status"])
        self.assertEqual([4, 6, 8], [case["generated_tokens"] for case in profile["cases"]])
        four = profile["cases"][0]
        self.assertEqual(3, four["decode_transitions"])
        self.assertEqual(
            8.0,
            four["raw_metrics"]["compile"]["value"],
        )
        self.assertEqual(
            2.0,
            four["normalized_metrics"]["per_generated_token"]["compile"]["value"],
        )
        self.assertEqual(
            8.0 / 3.0,
            four["normalized_metrics"]["per_decode_transition"]["compile"]["value"],
        )
        self.assertEqual(
            "second/generated_token",
            four["normalized_metrics"]["per_generated_token"]["compile"]["unit"],
        )
        self.assertIsNone(
            four["raw_metrics"]["available_memory_metrics"]["host_mem_available"]
        )
        self.assertIn("computer-local", profile["measurement_scope"])
        self.assertIn("not FPGA", profile["measurement_scope"])

    def test_missing_evidence_fails_closed(self) -> None:
        (self.cases[0][1] / "timing.json").unlink()
        with self.assertRaisesRegex(profiler.ProfileError, "sealed member is missing"):
            profiler.build_profile(self.cases)

    def test_missing_token_case_fails_closed(self) -> None:
        with self.assertRaisesRegex(profiler.ProfileError, "exactly three"):
            profiler.build_profile(self.cases[:2])

    def test_mixed_source_evidence_fails_closed(self) -> None:
        result_path = self.cases[1][2] / "result.json"
        result = json.loads(result_path.read_text())
        result["attempt"]["tree_root_sha256"] = "mixed"
        write_json(result_path, result)
        seal_oracle(self.cases[1][2])
        with self.assertRaisesRegex(profiler.ProfileError, "mixed-source"):
            profiler.build_profile(self.cases)

    def test_cross_case_source_binding_mismatch_fails_closed(self) -> None:
        _tokens, attempt, oracle = self.cases[1]
        different_source = hashlib.sha256(b"different-source").hexdigest()
        manifest_path = attempt / "attempt-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_and_rtl"]["sha256"] = different_source
        write_json(manifest_path, manifest)
        result_path = oracle / "result.json"
        result = json.loads(result_path.read_text())
        result["source_binding"]["sha256"] = different_source
        write_json(result_path, result)
        self.reseal_case(1)
        with self.assertRaisesRegex(
            profiler.ProfileError,
            "cohort is mixed-source",
        ):
            profiler.build_profile(self.cases)

    def test_fallback_enabled_evidence_fails_closed(self) -> None:
        summary_path = self.cases[2][1] / "runtime-output/run_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["software_transformer_or_logits_fallback"] = True
        write_json(summary_path, summary)
        self.reseal_case(2)
        with self.assertRaisesRegex(profiler.ProfileError, "fallback"):
            profiler.build_profile(self.cases)

    def test_mismatched_evidence_fails_closed(self) -> None:
        product_path = self.cases[0][1] / "runtime-output/product_result.json"
        product = json.loads(product_path.read_text())
        product["generated_token_ids"][-1] = 999
        write_json(product_path, product)
        self.reseal_case(0)
        with self.assertRaisesRegex(profiler.ProfileError, "token sequence mismatched"):
            profiler.build_profile(self.cases)

    def test_incomplete_evidence_fails_closed(self) -> None:
        result_path = self.cases[1][2] / "result.json"
        result = json.loads(result_path.read_text())
        result["agreement"]["retained_payload_bytes_compared"] = 0
        write_json(result_path, result)
        seal_oracle(self.cases[1][2])
        with self.assertRaisesRegex(profiler.ProfileError, "must be positive"):
            profiler.build_profile(self.cases)

    def test_cli_output_is_deterministic_and_refuses_replacement(self) -> None:
        outputs = [self.root / "profile-a.json", self.root / "profile-b.json"]
        base_command = ["python3", "-B", str(Path(profiler.__file__).resolve())]
        for tokens, attempt, oracle in self.cases:
            base_command.extend(["--case", str(tokens), str(attempt), str(oracle)])
        for output in outputs:
            completed = subprocess.run(
                [*base_command, "--output", str(output)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        completed = subprocess.run(
            [*base_command, "--output", str(outputs[0])],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(2, completed.returncode)
        self.assertIn("output already exists", completed.stderr)


if __name__ == "__main__":
    unittest.main()
