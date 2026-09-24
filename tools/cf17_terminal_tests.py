from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = load_module("_cf17_0002_runner_test", PACKAGE / "authority_runner.py")
validator = load_module("_cf17_0002_validator_test", PACKAGE / "validate_package.py")


def authority_for(state: Path) -> dict[str, object]:
    authority = json.loads((PACKAGE / "authority.json").read_text(encoding="utf-8"))
    authority["consumption_evidence_namespace"] = str(state)
    authority["output_namespace"] = str(state / "attempt-0001")
    argv = list(authority["rtl_endpoint_invocation"]["argv"])
    argv[argv.index("--output") + 1] = str(state / "attempt-0001/endpoint-output")
    authority["rtl_endpoint_invocation"]["argv"] = argv
    return authority


def write_success_commands(output: Path) -> None:
    host = runner.validate_host_evidence(
        json.loads((PACKAGE / "authority.json").read_text(encoding="utf-8"))
    )
    commands = output / "commands.jsonl"
    with commands.open("w", encoding="utf-8") as handle:
        ordinal = 0
        for position in host:
            for digest in position["per_layer_kv_digests"]:
                handle.write(
                    json.dumps(
                        {
                            "ordinal": ordinal,
                            "token_step": position["absolute_position"],
                            "operator": "kv_write",
                            "destination_sha256": digest,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                ordinal += 1
            for tile in range(4748):
                handle.write(
                    json.dumps(
                        {
                            "ordinal": ordinal,
                            "token_step": position["absolute_position"],
                            "operator": "lm_head_tile",
                            "destination_sha256": f"{tile:064x}",
                            "generated_token_after": position["selected_token_id"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                ordinal += 1


class PackageTests(unittest.TestCase):
    def test_package_closure_sources_evidence_and_zero_state(self) -> None:
        validator.validate_package(require_zero_state=True)

    def test_authority_is_endpoint_only_without_software_fallback(self) -> None:
        authority = json.loads(
            (PACKAGE / "authority.json").read_text(encoding="utf-8")
        )
        argv = authority["rtl_endpoint_invocation"]["argv"]
        self.assertEqual(authority["software_fallback"], "FORBIDDEN")
        self.assertFalse(any("product_probe.py" in value for value in argv))
        self.assertFalse(any("cf16-0002/authority_runner.py" in value for value in argv))
        self.assertEqual(
            authority["cf16_retry_replay_resume_relaunch"],
            "PERMANENTLY_FORBIDDEN",
        )


class TerminalAccountabilityTests(unittest.TestCase):
    def test_duplicate_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            authority = authority_for(state)
            runner.claim(authority, state=state, attempt=state / "attempt-0001")
            with self.assertRaisesRegex(
                runner.AuthorityError, "already or ambiguously consumed"
            ):
                runner.claim(authority, state=state, attempt=state / "attempt-0001")

    def test_success_publishes_endpoint_receipt_agreement_and_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            attempt = state / "attempt-0001"
            authority = authority_for(state)

            def successful(_argv, _environment, _timeout):
                write_success_commands(attempt / "endpoint-output")
                return {
                    "pid": 17,
                    "exit_code": 0,
                    "stdout": b"endpoint ok\n",
                    "stderr": b"",
                    "natural_terminal": True,
                }

            self.assertEqual(
                runner.execute_authorized(
                    authority,
                    state=state,
                    attempt=attempt,
                    process_runner=successful,
                ),
                0,
            )
            self.assertTrue((attempt / "endpoint-receipt.json").is_file())
            agreement = json.loads(
                (attempt / "host-rtl-agreement.json").read_text(encoding="utf-8")
            )
            self.assertEqual(agreement["status"], "PASS")
            self.assertTrue((state / "terminal-seal.json").is_file())

    def test_endpoint_process_failure_has_explicit_capture_and_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            attempt = state / "attempt-0001"
            authority = authority_for(state)

            def failed(_argv, _environment, _timeout):
                return {
                    "pid": 18,
                    "exit_code": 9,
                    "stdout": b"",
                    "stderr": b"endpoint failed\n",
                    "natural_terminal": True,
                }

            self.assertEqual(
                runner.execute_authorized(
                    authority,
                    state=state,
                    attempt=attempt,
                    process_runner=failed,
                ),
                2,
            )
            self.assertTrue((attempt / "endpoint-failure-capture.json").is_file())
            self.assertTrue((state / "terminal-seal.json").is_file())

    def test_endpoint_receipt_publication_failure_uses_failure_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            attempt = state / "attempt-0001"
            authority = authority_for(state)

            def successful(_argv, _environment, _timeout):
                write_success_commands(attempt / "endpoint-output")
                return {
                    "pid": 19,
                    "exit_code": 0,
                    "stdout": b"",
                    "stderr": b"",
                    "natural_terminal": True,
                }

            def fail_receipt(path: Path, value: object) -> None:
                if path.name == "endpoint-receipt.json":
                    raise OSError("injected endpoint receipt publication failure")
                runner.create_json(path, value)

            self.assertEqual(
                runner.execute_authorized(
                    authority,
                    state=state,
                    attempt=attempt,
                    process_runner=successful,
                    publisher=fail_receipt,
                ),
                2,
            )
            capture = json.loads(
                (attempt / "endpoint-failure-capture.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture["error_type"], "OSError")
            self.assertTrue((state / "terminal-seal.json").is_file())

    def test_primary_terminal_failure_uses_distinct_fsync_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            attempt = state / "attempt-0001"
            authority = authority_for(state)

            def failed(_argv, _environment, _timeout):
                return {
                    "pid": 20,
                    "exit_code": 3,
                    "stdout": b"",
                    "stderr": b"",
                    "natural_terminal": True,
                }

            def fail_primary_terminal(path: Path, value: object) -> None:
                if path == state / "terminal-seal.json":
                    raise OSError("injected primary terminal publication failure")
                runner.create_json(path, value)

            self.assertEqual(
                runner.execute_authorized(
                    authority,
                    state=state,
                    attempt=attempt,
                    process_runner=failed,
                    publisher=fail_primary_terminal,
                ),
                2,
            )
            fallback = state / "fallback/terminal-seal.json"
            self.assertTrue(fallback.is_file())
            seal = json.loads(fallback.read_text(encoding="utf-8"))
            self.assertEqual(seal["primary_publication_failure"], "OSError")


if __name__ == "__main__":
    unittest.main(verbosity=2)
