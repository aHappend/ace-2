#!/usr/bin/env python3
"""Static and synthetic-only tests for the closed Dynamic Scale32 baseline."""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import ace2_dynamic_scale32_baseline as baseline  # noqa: E402


class DynamicScale32BaselineExecutorTests(unittest.TestCase):
    maxDiff = None

    def test_01_static_surface_is_closed_and_candidate_free(self) -> None:
        path = TOOLS / "ace2_dynamic_scale32_baseline.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(
            any(
                name == "subprocess"
                or name.startswith("subprocess.")
                or name == "importlib"
                or name.startswith("importlib.")
                or "candidate" in name.lower()
                or "diagnostic" in name.lower()
                or "localize" in name.lower()
                or "discriminator" in name.lower()
                for name in imported
            )
        )
        banned_calls = {"__import__", "compile", "eval", "exec", "os.system"}
        observed_calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue(banned_calls.isdisjoint(observed_calls))
        self.assertNotIn("ace2_full_model_fixed_point", source)
        self.assertIn("require_distribution_versions(EXPECTED_SOFTWARE)", source)
        self.assertNotIn(".__version__", source)
        self.assertNotIn("--candidate", source)
        self.assertNotIn("--diagnostic", source)
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        candidate_strings = {value for value in strings if "candidate" in value.lower()}
        self.assertEqual(candidate_strings, {"candidate_model"})

    def test_02_cli_accepts_only_the_frozen_shape(self) -> None:
        parser = baseline.build_parser()
        parsed = parser.parse_args(
            ["--mode", "smoke", "--output-dir", "/tmp/out", "--smoke-token-limit", "8"]
        )
        self.assertEqual(parsed.mode, "smoke")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--mode",
                    "smoke",
                    "--output-dir",
                    "/tmp/out",
                    "--smoke-token-limit",
                    "8",
                    "--candidate-evidence",
                    "x",
                ]
            )

    def test_03_synthetic_fixture_emits_exact_deterministic_five(self) -> None:
        manifest, config = baseline.load_contracts(ROOT)
        mode_contract = baseline.validate_mode("smoke", 8)
        descriptor = {
            "dtype": "int8",
            "shape": [1, 1, 4],
            "sha256": "1" * 64,
            "minimum": -2.0,
            "maximum": 3.0,
            "squared_l2": 18.0,
        }
        payload = {
            "ordered_traces": [
                {
                    "dataset": "wikitext2",
                    "sequence_index": 0,
                    "input_ids": descriptor,
                    "traces": [
                        {
                            "ordinal": 0,
                            "operator": "layer_0.input_rmsnorm",
                            "tensor": descriptor,
                        },
                        {
                            "ordinal": 1,
                            "operator": "layer_0.v_proj",
                            "tensor": descriptor,
                        },
                    ],
                }
            ],
            "final_outputs": [
                {
                    "dataset": "wikitext2",
                    "sequence_index": 0,
                    "input_ids": descriptor,
                    "logits": descriptor,
                    "negative_log_likelihood": 1.25,
                    "scored_tokens": 3,
                    "predicted_token_ids": [1, 2, 3],
                }
            ],
            "metrics": {
                "wikitext2": {
                    "mean_negative_log_likelihood": 1.25,
                    "negative_log_likelihood": 3.75,
                    "perplexity": 3.4903429574618414,
                    "scored_tokens": 3,
                    "sequences": [],
                },
                "c4_en_512": {
                    "mean_negative_log_likelihood": 1.5,
                    "negative_log_likelihood": 4.5,
                    "perplexity": 4.4816890703380645,
                    "scored_tokens": 3,
                    "sequences": [],
                },
            },
            "observations": {
                "c4_calibration": {"record_count": 1, "record_sha256": "2" * 64},
                "wikitext2": {"record_count": 16, "record_sha256": "3" * 64},
                "c4_en_512": {"record_count": 1, "record_sha256": "4" * 64},
            },
            "resolved_model_revision": baseline.MODEL["revision"],
            "resolved_tokenizer_revision": baseline.MODEL["revision"],
            "software": dict(baseline.EXPECTED_SOFTWARE),
            "geometry": {
                "hidden_size": 896,
                "head_dim": 64,
                "num_attention_heads": 14,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
            },
        }
        environment = {
            "HOME": "/fixture/home",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": str(config["determinism"]["python_seed"]),
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": "/fixture/tmp",
            "TZ": "UTC",
            "XDG_CACHE_HOME": "/fixture/home/.cache",
        }
        with tempfile.TemporaryDirectory(prefix="ace2-baseline-executor-") as temporary:
            temporary_root = Path(temporary)
            first = temporary_root / "first"
            second = temporary_root / "second"
            with mock.patch.dict(os.environ, environment, clear=True):
                baseline.emit_outputs(first, mode_contract, ROOT, manifest, config, payload)
                baseline.emit_outputs(second, mode_contract, ROOT, manifest, config, payload)
            self.assertEqual(
                {path.name for path in first.iterdir()},
                set(baseline.OUTPUT_NAMES),
            )
            for name in baseline.OUTPUT_NAMES:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                value = json.loads((first / name).read_text(encoding="utf-8"))
                self.assertEqual(value["schema_version"], baseline.SCHEMA_VERSION)
                self.assertEqual(value["contract_id"], baseline.CONTRACT_ID)
            contract = json.loads((first / "run_contract.json").read_text(encoding="utf-8"))
            self.assertEqual(contract["output_schema"]["files"], list(baseline.OUTPUT_NAMES))
            self.assertEqual(contract["architecture_boundary"]["first_unsupported_layer_operator"], "layer_0.rope_q")
            self.assertEqual(contract["baseline_model"], 1)
            self.assertEqual(contract["candidate_model"], 0)

    def test_04_final_rmsnorm_to_lm_head_preserves_quantized_codes(self) -> None:
        import torch
        from torch import nn

        collector = baseline.TraceCollector(torch)
        runtime = baseline.make_fixed_runtime(torch, nn, {}, collector)
        norm_source = SimpleNamespace(weight=torch.ones(baseline.RMS_HIDDEN_SIZE))
        final_norm = runtime["FixedRMSNorm"](
            norm_source,
            input_scale=0.25,
            output_scale=0.5,
            operator="final_rmsnorm",
        )
        weight = torch.zeros((1, baseline.RMS_HIDDEN_SIZE))
        weight[0, 0] = 0.7
        linear_source = SimpleNamespace(
            weight=weight,
            bias=None,
            in_features=baseline.RMS_HIDDEN_SIZE,
            out_features=1,
        )
        lm_head = runtime["W4A8Linear"](
            linear_source,
            baseline.CalibrationRange(input_absmax=12.7, output_absmax=12.7),
        )
        self.assertNotEqual(lm_head.input_scale, final_norm.output_scale)
        lm_head.bind_quantized_input_scale(final_norm.output_scale)

        normalized_codes = final_norm(torch.ones((1, 1, baseline.RMS_HIDDEN_SIZE)))
        logits = lm_head(normalized_codes)

        self.assertTrue(torch.equal(normalized_codes, torch.full_like(normalized_codes, 2.0)))
        self.assertTrue(torch.allclose(logits, torch.tensor([[[0.7]]]), atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
