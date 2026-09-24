from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.ace2_chat_demo import (
    DEFAULT_OUTPUT,
    DEFAULT_SYSTEM_PROMPT,
    TOKENIZER_REPOSITORY,
    main,
    prepare_profile,
    read_conversation_file,
    read_prompt,
    tokenize_messages,
    tokenize_prompt,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.messages: list[dict[str, str]] = []
        self.template_args: tuple[bool, bool] | None = None
        self.applied_template: str | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encoded.append(text)
        return [100 + len(self.encoded), 200 + len(text.encode("utf-8"))]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
    ) -> list[int]:
        self.messages = [dict(message) for message in messages]
        self.template_args = (tokenize, add_generation_prompt)
        self.applied_template = chat_template
        return [1, 2, 3, 4]


class Ace2ChatHostInterfaceTest(unittest.TestCase):
    def test_prompt_file_preserves_exact_utf8_bytes(self) -> None:
        raw = b"  caf\xc3\xa9 \xce\xbb\r\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompt.txt"
            path.write_bytes(raw)
            prompt = read_prompt(None, path)

        self.assertEqual(prompt.encode("utf-8"), raw)
        tokenizer = RecordingTokenizer()
        metadata = tokenize_prompt(tokenizer, prompt, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(metadata["prompt_utf8_bytes"], len(raw))
        self.assertEqual(metadata["prompt_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(tokenizer.messages[-1]["content"].encode("utf-8"), raw)

    def test_conversation_file_preserves_message_order(self) -> None:
        expected = [
            {"role": "system", "content": "system café"},
            {"role": "user", "content": "first λ"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second 雪"},
        ]
        raw = json.dumps(
            {"messages": expected},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "conversation.json"
            path.write_bytes(raw)
            messages = read_conversation_file(
                path,
                default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            )

        self.assertEqual(messages, expected)
        tokenizer = RecordingTokenizer()
        metadata = tokenize_messages(tokenizer, messages, chat_template="pinned")
        self.assertEqual(tokenizer.messages, expected)
        self.assertEqual(tokenizer.template_args, (True, True))
        self.assertEqual(tokenizer.applied_template, "pinned")
        self.assertEqual(metadata["message_count"], 4)
        self.assertEqual(metadata["user_message_count"], 2)
        self.assertEqual(metadata["assistant_message_count"], 1)

    def test_repeated_interactive_turns_are_state_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "tools.ace2_chat_demo.sys.stdin.isatty",
            return_value=True,
        ), mock.patch(
            "builtins.input",
            side_effect=[" first turn ", "second turn", EOFError],
        ), mock.patch(
            "tools.ace2_chat_demo.run_request",
            return_value=0,
        ) as runner:
            self.assertEqual(main(["--output", temporary]), 0)

        self.assertEqual(runner.call_count, 2)
        first_args, first_messages, first_output = runner.call_args_list[0].args
        second_args, second_messages, second_output = runner.call_args_list[1].args
        self.assertIs(first_args, second_args)
        self.assertEqual(
            first_messages,
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "first turn"},
            ],
        )
        self.assertEqual(
            second_messages,
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "second turn"},
            ],
        )
        self.assertIsNot(first_messages, second_messages)
        self.assertEqual(first_output, Path(temporary).resolve() / "turn-0001")
        self.assertEqual(second_output, Path(temporary).resolve() / "turn-0002")

    def test_generated_token_bounds_accept_one_and_256(self) -> None:
        for value in (1, 256):
            with self.subTest(value=value), mock.patch(
                "tools.ace2_chat_demo.run_request",
                return_value=0,
            ) as runner:
                self.assertEqual(
                    main(["--prompt", "bounded", "--max-new-tokens", str(value)]),
                    0,
                )
                self.assertEqual(runner.call_args.args[0].max_new_tokens, value)

    def test_generated_token_bounds_reject_zero_and_257(self) -> None:
        for value in (0, 257):
            with self.subTest(value=value), mock.patch(
                "tools.ace2_chat_demo.run_request"
            ) as runner, self.assertRaisesRegex(
                RuntimeError,
                "--max-new-tokens must be between 1 and 256",
            ):
                main(["--prompt", "bounded", "--max-new-tokens", str(value)])
            runner.assert_not_called()

    def test_default_noninteractive_stdin_remains_one_shot_base(self) -> None:
        with mock.patch(
            "tools.ace2_chat_demo.sys.stdin.isatty",
            return_value=False,
        ), mock.patch(
            "tools.ace2_chat_demo.sys.stdin.read",
            return_value="  stdin default  \n",
        ), mock.patch(
            "tools.ace2_chat_demo.run_request",
            return_value=0,
        ) as runner:
            self.assertEqual(main([]), 0)

        runner.assert_called_once()
        args, messages, output = runner.call_args.args
        self.assertEqual(args.max_new_tokens, 4)
        self.assertFalse(args.diagnostic_instruct_prepare)
        self.assertEqual(prepare_profile(False)["repository"], TOKENIZER_REPOSITORY)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "stdin default"},
            ],
        )
        self.assertEqual(output, DEFAULT_OUTPUT)


if __name__ == "__main__":
    unittest.main()
