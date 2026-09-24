from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools import ace2_quality_contracts as quality
from tools import grouped_w4a8_payload_conformance as conformance
from tools import model_hardware_contract as hardware
from tools import qualify_stage1_w4a8_software as runtime
from tools import run_file_backed_w4a8_chat as command
from verification.test_grouped_w4a8_payload_conformance import (
    CompactFrozenTokenizer,
    CompactPayloadDecoderStackModel,
)


class FileBackedW4A8ChatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = copy.deepcopy(
            hardware.load_descriptor("qwen2.5-0.5b")
        )
        self.descriptor["dimensions"].update(
            {
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 32,
            }
        )
        self.prompt = "Explain cache reuse: 你好"
        unit_record = quality.pack_scale32(0x8000, 0)
        payloads = {}
        for index, name in enumerate(runtime.full_qwen_payload_names()):
            weights = [[0] * 32 for _ in range(32)]
            for output, row in enumerate(weights):
                row[(output + index) % 32] = -1 if index & 1 else 1
            payloads[name] = conformance.encode_payload(
                self.descriptor,
                32,
                weights,
                [[unit_record] for _ in range(32)],
            )
        self.package = conformance.encode_payload_package(
            payloads,
            runtime.full_qwen_payload_names(),
        )

    def runtime_factory(self, descriptor, package, contract):
        return self.runtime_factory_for_prompt(self.prompt)(
            descriptor,
            package,
            contract,
        )

    def runtime_factory_for_prompt(self, prompt):
        return self.runtime_factory_for_prompts([prompt])

    def runtime_factory_for_prompts(self, prompts):
        class BatchTokenizer(CompactFrozenTokenizer):
            def __init__(self, system, accepted_prompts):
                super().__init__(system, accepted_prompts[0])
                self.accepted_prompts = set(accepted_prompts)

            def apply_chat_template(self, messages, **kwargs):
                prompt = messages[1]["content"]
                if prompt not in self.accepted_prompts:
                    raise RuntimeError("frozen prompt differs")
                self.prompt = prompt
                return super().apply_chat_template(messages, **kwargs)

        def factory(descriptor, package, contract):
            model = CompactPayloadDecoderStackModel(descriptor)
            runtime.install_full_qwen_payload_package(model, package, descriptor)
            for layer in model.model.layers:
                layer.self_attn.rope_diagnostic_mechanism = (
                    command.evaluator.fixed.ACTIVE_ROPE_MECHANISM
                )
            command.evaluator.enable_w4a8_kv_cache(model)
            generation = contract["shared_w4a8_contract"]["generation"]
            return (
                model,
                BatchTokenizer(
                    generation["canonical_system_message"],
                    prompts,
                ),
            )

        return factory

    class ConversationTokenizer:
        def __init__(self, system):
            self.system = system
            self.render_count = 0

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            return_tensors,
        ):
            if messages[0] != {"role": "system", "content": self.system}:
                raise RuntimeError("conversation system message differs")
            if not tokenize or not add_generation_prompt or return_tensors != "pt":
                raise RuntimeError("conversation template options differ")
            encoded = bytearray(self.system.encode("utf-8"))
            expected_role = "user"
            for message in messages[1:]:
                if message["role"] != expected_role:
                    raise RuntimeError("conversation role ordering differs")
                if expected_role == "user":
                    encoded.extend(b"\x01")
                    encoded.extend(message["content"].encode("utf-8"))
                    expected_role = "assistant"
                else:
                    encoded.extend(message["content"].encode("utf-8"))
                    expected_role = "user"
            if expected_role != "assistant":
                raise RuntimeError("conversation must end with a user turn")
            self.render_count += 1
            return torch.tensor([list(encoded)], dtype=torch.long)

        def decode(
            self,
            token_ids,
            *,
            skip_special_tokens,
            clean_up_tokenization_spaces,
        ):
            if not skip_special_tokens or clean_up_tokenization_spaces:
                raise RuntimeError("conversation decode policy differs")
            return bytes(token_ids).decode("utf-8")

    def conversation_runtime_factory(self, tokenizer_type=None):
        tokenizer_type = tokenizer_type or self.ConversationTokenizer

        def factory(descriptor, package, contract):
            model = CompactPayloadDecoderStackModel(descriptor)
            runtime.install_full_qwen_payload_package(model, package, descriptor)
            for layer in model.model.layers:
                layer.self_attn.rope_diagnostic_mechanism = (
                    command.evaluator.fixed.ACTIVE_ROPE_MECHANISM
                )
            command.evaluator.enable_w4a8_kv_cache(model)
            system = contract["shared_w4a8_contract"]["generation"][
                "canonical_system_message"
            ]
            return model, tokenizer_type(system)

        return factory

    def write_package(self, directory: str) -> Path:
        package_path = Path(directory) / "model.ace2w4m1"
        package_path.write_bytes(self.package)
        return package_path

    def test_valid_package_emits_cacheful_machine_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package_path = self.write_package(temporary)
            output = command.execute(
                package_path,
                self.prompt,
                3,
                descriptor=self.descriptor,
                runtime_factory=self.runtime_factory,
            )

        self.assertEqual(output["status"], "PASS")
        generation = output["generation"]
        self.assertEqual(len(generation["generated_token_ids"]), 3)
        self.assertEqual(generation["termination_reason"], "max_new_tokens")
        self.assertEqual(
            len(generation["per_step_integer_logit_tensor_sha256"]),
            3,
        )
        self.assertTrue(generation["decoded_text"])
        for cache in generation["per_step_kv_cache"]:
            self.assertEqual(len(cache["layers"]), 24)
            self.assertEqual(
                [layer["layer_index"] for layer in cache["layers"]],
                list(range(24)),
            )
            for layer in cache["layers"]:
                self.assertEqual(
                    set(layer),
                    {
                        "key_scale32_sha256",
                        "key_sha256",
                        "layer_index",
                        "sequence_length",
                        "value_sha256",
                    },
                )

    def test_batch_reuses_runtime_preserves_order_and_resets_cache(self) -> None:
        prompts = [
            {"prompt_id": "zh-cache", "user": "解释缓存：你好"},
            {"prompt_id": "jp-cache", "user": "キャッシュを説明して"},
        ]
        constructions = 0

        def counted_factory(descriptor, package, contract):
            nonlocal constructions
            constructions += 1
            return self.runtime_factory_for_prompts(
                [prompt["user"] for prompt in prompts]
            )(descriptor, package, contract)

        with tempfile.TemporaryDirectory() as temporary:
            package_path = self.write_package(temporary)
            batch = command.execute_batch(
                package_path,
                prompts,
                3,
                descriptor=self.descriptor,
                runtime_factory=counted_factory,
            )
            isolated = [
                command.execute_batch(
                    package_path,
                    [prompt],
                    3,
                    descriptor=self.descriptor,
                    runtime_factory=self.runtime_factory_for_prompt(
                        prompt["user"]
                    ),
                )["generations"][0]
                for prompt in prompts
            ]

        self.assertEqual(constructions, 1)
        self.assertEqual(batch["prompt_count"], 2)
        self.assertEqual(
            [result["prompt_id"] for result in batch["generations"]],
            ["zh-cache", "jp-cache"],
        )
        self.assertEqual(batch["generations"], isolated)
        generation_contract = command.load_contract()[0][
            "shared_w4a8_contract"
        ]["generation"]
        for prompt, result in zip(prompts, batch["generations"], strict=True):
            initial_length = len(
                (
                    generation_contract["canonical_system_message"]
                    + "\0"
                    + prompt["user"]
                ).encode("utf-8")
            )
            self.assertTrue(result["per_step_kv_cache"])
            for layer in result["per_step_kv_cache"][0]["layers"]:
                self.assertEqual(layer["sequence_length"], initial_length)

    def test_invalid_batches_fail_before_runtime_build(self) -> None:
        invalid_batches = {
            "not-list": {},
            "empty": [],
            "non-object": ["prompt"],
            "missing-field": [{"prompt_id": "one"}],
            "extra-field": [
                {"prompt_id": "one", "user": "hello", "extra": True}
            ],
            "empty-id": [{"prompt_id": "", "user": "hello"}],
            "empty-user": [{"prompt_id": "one", "user": ""}],
            "duplicate-id": [
                {"prompt_id": "one", "user": "hello"},
                {"prompt_id": "one", "user": "different"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            package_path = self.write_package(temporary)

            def forbidden_factory(*_args):
                self.fail("runtime construction began for an invalid batch")

            for name, prompts in invalid_batches.items():
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    command.execute_batch(
                        package_path,
                        prompts,
                        2,
                        descriptor=self.descriptor,
                        runtime_factory=forbidden_factory,
                    )

    def test_batch_reader_requires_strict_utf8_json_and_unique_ids(self) -> None:
        cases = {
            "invalid-utf8": b"\xff",
            "invalid-json": b"[",
            "duplicate-member": (
                b'[{"prompt_id":"one","prompt_id":"two","user":"hello"}]'
            ),
            "non-standard-number": (
                b'[{"prompt_id":"one","user":"hello","value":NaN}]'
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            batch_path = Path(temporary) / "prompts.json"
            for name, payload in cases.items():
                with self.subTest(name=name):
                    batch_path.write_bytes(payload)
                    with self.assertRaises((RuntimeError, ValueError)):
                        command.read_prompt_batch(batch_path)

            expected = [
                {"prompt_id": "first", "user": "你好"},
                {"prompt_id": "second", "user": "مرحبا"},
            ]
            batch_path.write_text(
                json.dumps(expected, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(command.read_prompt_batch(batch_path), expected)

    def test_conversation_matches_fresh_full_prefix_and_grows_all_caches(
        self,
    ) -> None:
        turns = [
            {"turn_id": "first", "user": "你好"},
            {"turn_id": "second", "user": "continue"},
        ]
        constructions = 0

        def counted_factory(descriptor, package, contract):
            nonlocal constructions
            constructions += 1
            return self.conversation_runtime_factory()(
                descriptor,
                package,
                contract,
            )

        with tempfile.TemporaryDirectory() as temporary:
            package_path = self.write_package(temporary)
            output = command.execute_conversation(
                package_path,
                turns,
                2,
                descriptor=self.descriptor,
                runtime_factory=counted_factory,
            )
            package = package_path.read_bytes()

        self.assertEqual(constructions, 1)
        self.assertEqual(output["turn_count"], 2)
        contract = command.load_contract()[0]
        generation = copy.deepcopy(
            contract["shared_w4a8_contract"]["generation"]
        )
        generation["max_new_tokens"] = 2
        messages = [
            {
                "role": "system",
                "content": generation["canonical_system_message"],
            }
        ]
        for turn, incremental in zip(turns, output["turns"], strict=True):
            messages.append({"role": "user", "content": turn["user"]})
            model, tokenizer = self.conversation_runtime_factory()(
                self.descriptor,
                package,
                contract,
            )
            input_ids = command.evaluator.render_conversation_ids(
                tokenizer,
                messages,
            )
            fresh, _state = command.evaluator.generate_w4a8_continuation(
                model,
                tokenizer,
                turn["turn_id"],
                input_ids,
                generation,
            )
            for field in (
                "generated_token_ids",
                "per_step_integer_logit_tensor_sha256",
                "per_step_kv_cache",
            ):
                self.assertEqual(incremental[field], fresh[field])
            for step, cache in enumerate(incremental["per_step_kv_cache"]):
                self.assertTrue(
                    all(
                        layer["sequence_length"]
                        == len(incremental["input_token_ids"]) + step
                        for layer in cache["layers"]
                    )
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": incremental["decoded_text"],
                }
            )

    def test_conversation_termination_boundary_is_consumed_once(self) -> None:
        turns = [
            {"turn_id": "first", "user": "a"},
            {"turn_id": "second", "user": "b"},
        ]
        contract = command.load_contract()[0]
        generation = copy.deepcopy(
            contract["shared_w4a8_contract"]["generation"]
        )
        generation["max_new_tokens"] = 3
        generation["termination_token_ids"] = [0]
        model, tokenizer = self.conversation_runtime_factory()(
            self.descriptor,
            self.package,
            contract,
        )

        results = command.generate_conversation(
            model,
            tokenizer,
            turns,
            generation,
        )

        self.assertEqual(
            [result["termination_reason"] for result in results],
            ["termination_token_id:0", "termination_token_id:0"],
        )
        self.assertEqual(
            results[1]["input_token_ids"][
                len(results[0]["input_token_ids"]):
                len(results[0]["input_token_ids"]) + 1
            ],
            [0],
        )

    def test_conversation_prefix_template_drift_is_rejected(self) -> None:
        class DriftingTokenizer(self.ConversationTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                rendered = super().apply_chat_template(*args, **kwargs)
                if self.render_count == 2:
                    rendered[0, 0] += 1
                return rendered

        contract = command.load_contract()[0]
        generation = copy.deepcopy(
            contract["shared_w4a8_contract"]["generation"]
        )
        generation["max_new_tokens"] = 1
        model, tokenizer = self.conversation_runtime_factory(
            DriftingTokenizer
        )(self.descriptor, self.package, contract)

        with self.assertRaisesRegex(RuntimeError, "prefix/template drift"):
            command.generate_conversation(
                model,
                tokenizer,
                [
                    {"turn_id": "first", "user": "a"},
                    {"turn_id": "second", "user": "b"},
                ],
                generation,
            )

    def test_invalid_conversations_fail_before_runtime_build(self) -> None:
        invalid = {
            "not-list": {},
            "empty": [],
            "missing-field": [{"turn_id": "one"}],
            "extra-field": [{"turn_id": "one", "user": "a", "extra": 1}],
            "empty-id": [{"turn_id": "", "user": "a"}],
            "empty-user": [{"turn_id": "one", "user": ""}],
            "duplicate-id": [
                {"turn_id": "one", "user": "a"},
                {"turn_id": "one", "user": "b"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            package_path = self.write_package(temporary)

            def forbidden_factory(*_args):
                self.fail("runtime construction began for invalid conversation")

            for name, turns in invalid.items():
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    command.execute_conversation(
                        package_path,
                        turns,
                        1,
                        descriptor=self.descriptor,
                        runtime_factory=forbidden_factory,
                    )

    def test_conversation_reader_is_strict_utf8_json(self) -> None:
        cases = {
            "invalid-utf8": b"\xff",
            "invalid-json": b"[",
            "duplicate-member": (
                b'[{"turn_id":"one","turn_id":"two","user":"hello"}]'
            ),
            "non-standard-number": (
                b'[{"turn_id":"one","user":"hello","value":NaN}]'
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "conversation.json"
            for name, payload in cases.items():
                with self.subTest(name=name):
                    path.write_bytes(payload)
                    with self.assertRaises((RuntimeError, ValueError)):
                        command.read_conversation(path)


    def test_malformed_and_truncated_packages_fail_before_runtime_build(self) -> None:
        malformed_packages = {
            "truncated": self.package[:-1],
            "trailing": self.package + b"\0",
            "bad-magic": b"BROKEN!!" + self.package[8:],
        }
        for name, package in malformed_packages.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                package_path = Path(temporary) / "model.ace2w4m1"
                package_path.write_bytes(package)

                def forbidden_factory(*_args):
                    self.fail("runtime construction began for an invalid package")

                with self.assertRaises(conformance.PayloadError):
                    command.execute(
                        package_path,
                        self.prompt,
                        2,
                        descriptor=self.descriptor,
                        runtime_factory=forbidden_factory,
                    )


if __name__ == "__main__":
    unittest.main()
