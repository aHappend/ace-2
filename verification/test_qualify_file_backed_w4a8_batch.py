from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools import qualify_file_backed_w4a8_batch as qualification


def generation(prompt_id: str, seed: int, max_new_tokens: int) -> dict:
    input_ids = [seed, seed + 1, seed + 2]
    caches = []
    for step in range(max_new_tokens):
        caches.append(
            {
                "layers": [
                    {
                        "key_scale32_sha256": f"scale-{seed}-{step}-{layer}",
                        "key_sha256": f"key-{seed}-{step}-{layer}",
                        "layer_index": layer,
                        "sequence_length": len(input_ids) + step,
                        "value_sha256": f"value-{seed}-{step}-{layer}",
                    }
                    for layer in range(24)
                ],
                "sha256": f"cache-{seed}-{step}",
            }
        )
    return {
        "decoded_text": f"answer-{seed}",
        "decoded_text_sha256": f"text-{seed}",
        "generated_token_ids": list(range(seed, seed + max_new_tokens)),
        "generated_token_ids_sha256": f"tokens-{seed}",
        "input_token_ids": input_ids,
        "per_step_integer_logit_tensor_sha256": [
            f"logits-{seed}-{step}" for step in range(max_new_tokens)
        ],
        "per_step_kv_cache": caches,
        "per_step_kv_cache_sha256": [
            f"cache-{seed}-{step}" for step in range(max_new_tokens)
        ],
        "prompt_id": prompt_id,
        "termination_reason": "max_new_tokens",
        "use_cache": True,
    }


class QualifyFileBackedW4A8BatchTest(unittest.TestCase):
    def test_qualification_compares_processes_and_preflights_malformed_input(
        self,
    ) -> None:
        prompts = qualification.PROMPTS
        by_text = {
            prompt["user"]: generation(prompt["prompt_id"], index + 10, 2)
            for index, prompt in enumerate(prompts)
        }
        metadata = {
            "contract_sha256": "contract",
            "model_id": "qwen2.5-0.5b-instruct",
            "package_sha256": "package",
            "runtime": "fresh-pinned-cpu-fixed-w4a8",
            "schema_version": 1,
            "status": "PASS",
        }
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            if "--prompt-batch-file" in argv:
                path = Path(argv[argv.index("--prompt-batch-file") + 1])
                batch = json.loads(path.read_text(encoding="utf-8"))
                if batch[0]["prompt_id"] == "duplicate":
                    return subprocess.CompletedProcess(
                        argv,
                        2,
                        "",
                        "ACE2_W4A8_CHAT_ERROR: duplicate prompt_id in prompt batch: duplicate\n",
                    )
                output = {
                    **metadata,
                    "generations": [
                        by_text[prompt["user"]] for prompt in batch
                    ],
                    "prompt_count": len(batch),
                }
            else:
                path = Path(argv[argv.index("--prompt-file") + 1])
                record = dict(by_text[path.read_text(encoding="utf-8")])
                record["prompt_id"] = "stdin-or-file"
                output = {**metadata, "generation": record}
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps(output) + "\n",
                "",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "model.ace2w4m1"
            package.write_bytes(b"pinned-package")
            digest = hashlib.sha256(package.read_bytes()).hexdigest()
            output = root / "qualification"
            result = qualification.qualify(
                package,
                output,
                digest,
                2,
                60,
                runner=runner,
            )
            persisted = json.loads(
                (output / "qualification.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result, persisted)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][1]["env"]["PYTHONHASHSEED"], "20260729")
        malformed_package = calls[-1][0][
            calls[-1][0].index("--package") + 1
        ]
        self.assertTrue(malformed_package.endswith("must-not-be-opened.ace2w4m1"))

    def test_cache_length_contamination_is_rejected(self) -> None:
        record = generation("prompt", 5, 2)
        record["per_step_kv_cache"][0]["layers"][0]["sequence_length"] += 1
        with self.assertRaisesRegex(RuntimeError, "fresh prompt state"):
            qualification.validate_fresh_cache(record, "prompt")


if __name__ == "__main__":
    unittest.main()
