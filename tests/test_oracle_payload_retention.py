from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts import verify_full_chain_independent_oracle as oracle
from tools import rtl_arbitrary_text_generation_backend as backend


class OraclePayloadRetentionTest(unittest.TestCase):
    def setUp(self) -> None:
        verification_root = backend.ROOT / "reports" / "verification"
        verification_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            dir=verification_root,
            prefix="payload-retention-test-",
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_payloads(
        self,
        execution: Path,
        payloads: dict[str, bytes],
    ) -> dict[str, dict[str, object]]:
        tensors = execution / "tensors"
        tensors.mkdir(parents=True)
        records = {
            name: backend.write_binary(tensors / name, payload)
            for name, payload in payloads.items()
        }
        backend.write_binary(tensors / "transient.bin", b"discard")
        (execution / "vectors").mkdir()
        backend.write_binary(execution / "vectors/transient.hex", b"00\n")
        return records

    def _write_whole_sequence_fixture(
        self,
        name: str,
    ) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
        fixture_root = self.root / name
        prompt_token_ids = [11]
        generated_token_ids = [21, 22, 23]
        input_token_ids = prompt_token_ids + generated_token_ids[:-1]
        reference_positions = []
        summary_positions = []
        rtl_paths = []
        full_k = bytearray()
        full_v = bytearray()

        for position, input_token_id in enumerate(input_token_ids):
            position_dir = fixture_root / f"position-{position}"
            k_append = bytes([position + 1])
            v_append = bytes([position + 11])
            full_k.extend(k_append)
            full_v.extend(v_append)
            layer_output = bytes([position + 31])
            layer_output_scale = struct.pack("<d", float(position + 1))
            records = self._write_payloads(
                position_dir,
                {
                    "k_append_s8.bin": k_append,
                    "v_append_s8.bin": v_append,
                    "full_k_cache_s8.bin": bytes(full_k),
                    "full_v_cache_s8.bin": bytes(full_v),
                    "layer_output_s8.bin": layer_output,
                    "layer_output_scale_f64le.bin": layer_output_scale,
                },
            )
            rtl = {
                "kv_cache": {
                    "rtl_observed_k": records["k_append_s8.bin"],
                    "rtl_observed_v": records["v_append_s8.bin"],
                    "full_k_cache_sha256": records["full_k_cache_s8.bin"]["sha256"],
                    "full_v_cache_sha256": records["full_v_cache_s8.bin"]["sha256"],
                },
                "tensors": {
                    "full_k_cache_s8.bin": records["full_k_cache_s8.bin"],
                    "full_v_cache_s8.bin": records["full_v_cache_s8.bin"],
                    "layer_output_s8.bin": records["layer_output_s8.bin"],
                    "layer_output_scale_f64le.bin": records[
                        "layer_output_scale_f64le.bin"
                    ],
                },
                "final_residual": {
                    "output_sha256": records["layer_output_s8.bin"]["sha256"]
                },
            }
            rtl_path = position_dir / "rtl_execution.json"
            backend.write_json(rtl_path, rtl)
            rtl_paths.append(rtl_path)
            rtl_record = backend.file_record(rtl_path)
            phase = "prefill" if position == 0 else "decode"
            reference_positions.append(
                {
                    "absolute_position": position,
                    "phase": phase,
                    "input_token_id": input_token_id,
                    "layers": [
                        {
                            "layer_id": layer_id,
                            "k_append": k_append,
                            "v_append": v_append,
                            "full_k_cache": bytes(full_k),
                            "full_v_cache": bytes(full_v),
                            "layer_output": layer_output,
                            "layer_output_scale": layer_output_scale,
                            "rank1": None,
                        }
                        for layer_id in range(oracle.EXPECTED_LAYERS)
                    ],
                }
            )
            summary_positions.append(
                {
                    "absolute_position": position,
                    "phase": phase,
                    "input_token_id": input_token_id,
                    "layers": [
                        {
                            "layer_id": layer_id,
                            "rtl_execution": rtl_record,
                        }
                        for layer_id in range(oracle.EXPECTED_LAYERS)
                    ],
                }
            )

        reference_heads = []
        summary_heads = []
        for generation_index, selected_token_id in enumerate(generated_token_ids):
            head_dir = fixture_root / f"head-{generation_index}"
            final_rmsnorm = bytes([generation_index + 41])
            final_rmsnorm_scale = struct.pack("<d", float(generation_index + 4))
            logits = bytes([generation_index + 51, generation_index + 52])
            logit_scale = struct.pack("<d", float(generation_index + 7))
            selected_logit = generation_index + 9
            records = {
                artifact.removesuffix(".bin"): record
                for artifact, record in self._write_payloads(
                    head_dir,
                    {
                        "final_rmsnorm_s8.bin": final_rmsnorm,
                        "final_rmsnorm_scale_f64le.bin": final_rmsnorm_scale,
                        "lm_head_output_s8.bin": logits,
                        "lm_head_output_scale_f64le.bin": logit_scale,
                    },
                ).items()
            }
            head = {
                "final_rmsnorm": {
                    "output_sha256": records["final_rmsnorm_s8"]["sha256"]
                },
                "lm_head": {
                    "artifacts": records,
                    "output_sha256": records["lm_head_output_s8"]["sha256"],
                    "top_token": selected_token_id,
                    "top_logit_s8": selected_logit,
                    "saturation_count": 0,
                    "accumulator_min": -3,
                    "accumulator_max": 5,
                },
            }
            head_path = head_dir / "head_execution.json"
            backend.write_json(head_path, head)
            reference_heads.append(
                {
                    "generation_index": generation_index,
                    "source_absolute_position": generation_index,
                    "final_rmsnorm": final_rmsnorm,
                    "final_rmsnorm_scale": final_rmsnorm_scale,
                    "logits": logits,
                    "logit_scale": logit_scale,
                    "selected_token_id": selected_token_id,
                    "selected_logit_s8": selected_logit,
                    "saturation_count": 0,
                    "accumulator_min": -3,
                    "accumulator_max": 5,
                }
            )
            summary_heads.append(
                {
                    "generation_index": generation_index,
                    "source_absolute_position": generation_index,
                    "head_execution": backend.file_record(head_path),
                    "selected_token_id": selected_token_id,
                    "selected_logit_s8": selected_logit,
                }
            )

        common = {
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_token_ids,
            "decoded_text": "synthetic continuation",
        }
        reference = {
            **common,
            "positions": reference_positions,
            "heads": reference_heads,
        }
        summary = {
            **common,
            "token_executions": summary_positions,
            "head_steps": summary_heads,
        }
        return reference, summary, rtl_paths

    def test_strict_oracle_payloads_survive_pruning_and_match_bytes(self) -> None:
        layer_dir = self.root / "layer"
        layer_payloads = {
            "layer_output_s8.bin": b"\x80\x00\x7f",
            "layer_output_scale_f64le.bin": b"\x00\x00\x00\x00\x00\x00\xf0?",
        }
        layer_records = self._write_payloads(layer_dir, layer_payloads)
        layer_persistence = backend.prune_transient_execution_artifacts(layer_dir)
        layer_result = {"tensors": layer_records}
        backend.write_json(layer_dir / "rtl_execution.json", layer_result)

        head_dir = self.root / "head"
        head_payloads = {
            "final_rmsnorm_s8.bin": b"\x81\x01\x7e",
            "final_rmsnorm_scale_f64le.bin": b"\x00\x00\x00\x00\x00\x00\x00@",
            "lm_head_output_s8.bin": bytes(range(16)),
            "lm_head_output_scale_f64le.bin": b"\x00\x00\x00\x00\x00\x00\x08@",
        }
        head_records = {
            name.removesuffix(".bin"): record
            for name, record in self._write_payloads(head_dir, head_payloads).items()
        }
        head_persistence = backend.prune_transient_execution_artifacts(head_dir)
        head_result = {"lm_head": {"artifacts": head_records}}
        backend.write_json(head_dir / "head_execution.json", head_result)

        summary = {
            "token_executions": [
                {
                    "layers": [
                        {
                            "rtl_execution": backend.file_record(
                                layer_dir / "rtl_execution.json"
                            )
                        }
                    ]
                }
            ],
            "head_steps": [
                {
                    "head_execution": backend.file_record(
                        head_dir / "head_execution.json"
                    )
                }
            ],
        }
        checked = oracle.verify_retained_comparison_payloads(summary)

        self.assertEqual(6, checked["records_checked"])
        self.assertEqual(
            sum(map(len, layer_payloads.values()))
            + sum(map(len, head_payloads.values())),
            checked["bytes_checked"],
        )
        self.assertEqual(2, layer_persistence["retained_tensor_files"])
        self.assertEqual(4, head_persistence["retained_tensor_files"])
        self.assertFalse((layer_dir / "tensors/transient.bin").exists())
        self.assertFalse((head_dir / "vectors").exists())
        for name, payload in layer_payloads.items():
            oracle.require_payload_matches_record(
                payload,
                layer_records[name],
                name,
            )
        for name, payload in head_payloads.items():
            oracle.require_payload_matches_record(
                payload,
                head_records[name.removesuffix(".bin")],
                name,
            )

    def test_missing_strict_oracle_payload_fails_closed(self) -> None:
        execution = self.root / "missing"
        payloads = {
            "layer_output_s8.bin": b"\x00",
            "layer_output_scale_f64le.bin": b"\x00" * 8,
        }
        records = self._write_payloads(execution, payloads)
        backend.prune_transient_execution_artifacts(execution)
        (execution / "tensors/layer_output_s8.bin").unlink()
        backend.write_json(
            execution / "rtl_execution.json",
            {"tensors": records},
        )
        summary = {
            "token_executions": [
                {
                    "layers": [
                        {
                            "rtl_execution": backend.file_record(
                                execution / "rtl_execution.json"
                            )
                        }
                    ]
                }
            ],
            "head_steps": [],
        }

        with self.assertRaisesRegex(
            oracle.sealed_verifier.VerificationError,
            "recorded artifact is missing: .*layer_output_s8.bin",
        ):
            oracle.verify_retained_comparison_payloads(summary)

    def test_retained_kv_value_mutations_fail_whole_sequence_comparison(
        self,
    ) -> None:
        mutations = (
            ("rtl_observed_k", "K"),
            ("rtl_observed_v", "V"),
        )
        for record_name, label in mutations:
            with self.subTest(cache=label):
                reference, summary, rtl_paths = self._write_whole_sequence_fixture(
                    label.lower()
                )
                agreement = oracle.compare_reference(reference, summary)
                self.assertEqual(3, agreement["positions_checked"])
                self.assertEqual(72, agreement["layers_checked"])
                self.assertEqual(3, agreement["full_vocabulary_head_steps_checked"])

                rtl_path = rtl_paths[1]
                rtl = oracle.sealed_verifier.load_json(rtl_path)
                record = rtl["kv_cache"][record_name]
                retained_path = oracle.sealed_verifier.project_path(record["path"])
                payload = retained_path.read_bytes()
                retained_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                rtl["kv_cache"][record_name] = backend.file_record(retained_path)
                backend.write_json(rtl_path, rtl)
                rtl_record = backend.file_record(rtl_path)
                for layer in summary["token_executions"][1]["layers"]:
                    layer["rtl_execution"] = rtl_record

                with self.assertRaisesRegex(
                    oracle.OracleError,
                    f"independent {label} append differs",
                ):
                    oracle.compare_reference(reference, summary)

    def test_selected_token_mutation_fails_whole_sequence_comparison(self) -> None:
        reference, summary, _rtl_paths = self._write_whole_sequence_fixture("token")
        agreement = oracle.compare_reference(reference, summary)
        self.assertEqual(3, agreement["positions_checked"])
        self.assertEqual(["prefill", "decode", "decode"], [
            position["phase"] for position in reference["positions"]
        ])

        summary["head_steps"][1]["selected_token_id"] += 1
        with self.assertRaisesRegex(
            oracle.OracleError,
            "independent greedy token differs",
        ):
            oracle.compare_reference(reference, summary)


if __name__ == "__main__":
    unittest.main()
