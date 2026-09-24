#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BASE_CONSTRUCTOR = TOOLS / "construct_cf17_terminal_accountability.py"
BASE_RUNNER = TOOLS / "cf17_terminal_authority_runner.py"
BASE_VALIDATOR = TOOLS / "cf17_terminal_validate.py"
BASE_TESTS = TOOLS / "cf17_terminal_tests.py"
OLD_IDENTITY = "stage1-w4a8-r6-generation-probe-terminal-accountability-cf17-0002"
NEW_IDENTITY = "stage1-w4a8-r6-generation-probe-terminal-accountability-cf17-0003"


def replace_exact(text: str, old: str, new: str, description: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"cannot uniquely apply CF17 successor change: {description}")
    return text.replace(old, new)


def successor_runner() -> str:
    text = BASE_RUNNER.read_text(encoding="utf-8").replace(OLD_IDENTITY, NEW_IDENTITY)
    text = replace_exact(
        text,
        "Publisher = Callable[[Path, object], None]\n"
        "ProcessRunner = Callable[[list[str], dict[str, str], int], dict[str, Any]]\n",
        "Publisher = Callable[[Path, object], None]\n"
        "ProcessRunner = Callable[[list[str], dict[str, str], int], dict[str, Any]]\n"
        "PostConsumptionHook = Callable[[], None]\n",
        "post-consumption hook type",
    )
    text = replace_exact(
        text,
        """    mkdir_create_only(state)
    create_json(
        state / "authorization-consumed.json",
""",
        """    mkdir_create_only(state)
    mkdir_create_only(attempt)
    mkdir_create_only(attempt / "endpoint-output")
    mkdir_create_only(attempt / "fallback")
    mkdir_create_only(state / "fallback")
    create_json(
        state / "authorization-consumed.json",
""",
        "pre-consumption terminal namespaces",
    )
    text = replace_exact(
        text,
        """    create_json(
        state / "execution-claim.json",
        {
            "schema": "ace2-r6-cf17-execution-claim-v1",
            "status": "CLAIMED_AFTER_DURABLE_CONSUMPTION",
            "authorization_consumed_sha256": sha256_file(
                state / "authorization-consumed.json"
            ),
            "output_namespace": str(attempt),
        },
    )
    mkdir_create_only(attempt)
    mkdir_create_only(attempt / "endpoint-output")
    mkdir_create_only(attempt / "fallback")
    mkdir_create_only(state / "fallback")
    create_json(
        state / "consumption-receipt.json",
        {
            "schema": "ace2-r6-cf17-consumption-receipt-v1",
            "status": "DURABLE_BEFORE_ENDPOINT_PROCESS",
            "authorization_consumed_sha256": sha256_file(
                state / "authorization-consumed.json"
            ),
            "execution_claim_sha256": sha256_file(state / "execution-claim.json"),
        },
    )
""",
        "",
        "remove unaccounted post-consumption claim work",
    )
    text = replace_exact(
        text,
        """    process_runner: ProcessRunner = run_endpoint,
    publisher: Publisher = create_json,
) -> int:
    host_positions = validate_host_evidence(authority)
    claim(authority, state=state, attempt=attempt)
    create_json(
        attempt / "host-input-receipt.json",
        {
            "schema": "ace2-r6-cf17-host-input-receipt-v1",
            "status": "CF16_HOST_CAPTURE_AUTHENTICATED_READ_ONLY_BEFORE_ENDPOINT",
            "path": str(HOST_EVIDENCE),
            "sha256": sha256_file(HOST_EVIDENCE),
            "position_count": 4,
            "layer_state_count_per_position": 24,
            "cf16_authority_state_mutation": "FORBIDDEN",
        },
    )
    process: dict[str, Any] | None = None
    outcome: Path | None = None
    classification = "CF17_RUNNER_FAILURE"
    try:
        endpoint = authority["rtl_endpoint_invocation"]
""",
        """    process_runner: ProcessRunner = run_endpoint,
    publisher: Publisher = create_json,
    post_consumption_hook: PostConsumptionHook | None = None,
) -> int:
    host_positions = validate_host_evidence(authority)
    process: dict[str, Any] | None = None
    outcome: Path | None = None
    classification = "CF17_RUNNER_FAILURE"
    claim(authority, state=state, attempt=attempt)
    try:
        if post_consumption_hook is not None:
            post_consumption_hook()
        create_json(
            state / "execution-claim.json",
            {
                "schema": "ace2-r6-cf17-execution-claim-v1",
                "status": "CLAIMED_AFTER_DURABLE_CONSUMPTION",
                "authorization_consumed_sha256": sha256_file(
                    state / "authorization-consumed.json"
                ),
                "output_namespace": str(attempt),
            },
        )
        create_json(
            state / "consumption-receipt.json",
            {
                "schema": "ace2-r6-cf17-consumption-receipt-v1",
                "status": "DURABLE_BEFORE_ENDPOINT_PROCESS",
                "authorization_consumed_sha256": sha256_file(
                    state / "authorization-consumed.json"
                ),
                "execution_claim_sha256": sha256_file(
                    state / "execution-claim.json"
                ),
            },
        )
        create_json(
            attempt / "host-input-receipt.json",
            {
                "schema": "ace2-r6-cf17-host-input-receipt-v1",
                "status": "CF16_HOST_CAPTURE_AUTHENTICATED_READ_ONLY_BEFORE_ENDPOINT",
                "path": str(HOST_EVIDENCE),
                "sha256": sha256_file(HOST_EVIDENCE),
                "position_count": 4,
                "layer_state_count_per_position": 24,
                "cf16_authority_state_mutation": "FORBIDDEN",
            },
        )
        endpoint = authority["rtl_endpoint_invocation"]
""",
        "terminal-accountability boundary",
    )
    return text


def successor_validator() -> str:
    text = BASE_VALIDATOR.read_text(encoding="utf-8").replace(
        OLD_IDENTITY, NEW_IDENTITY
    )
    text = replace_exact(
        text,
        '    endpoint = authority.get("rtl_endpoint_invocation", {})\n'
        '    argv = endpoint.get("argv", [])\n',
        '    endpoint = authority.get("rtl_endpoint_invocation", {})\n'
        '    publication = authority.get("post_consumption_publication", {})\n'
        '    argv = endpoint.get("argv", [])\n',
        "publication contract load",
    )
    text = replace_exact(
        text,
        """        or contract.get("terminal_publication")
        != "PRIMARY_OR_DISTINCT_FSYNCED_FALLBACK_REQUIRED"
""",
        """        or publication.get("accountability_start")
        != "IMMEDIATELY_AFTER_DURABLE_AUTHORIZATION_CONSUMPTION"
        or contract.get("post_consumption_accountability")
        != "IMMEDIATELY_AFTER_DURABLE_AUTHORIZATION_CONSUMPTION"
        or contract.get("terminal_publication")
        != "PRIMARY_OR_DISTINCT_FSYNCED_FALLBACK_REQUIRED"
""",
        "immediate accountability validation",
    )
    return text


def successor_tests() -> str:
    text = BASE_TESTS.read_text(encoding="utf-8").replace(
        "_cf17_0002_", "_cf17_0003_"
    )
    test = '''
    def test_immediate_post_consumption_failure_is_captured_and_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            attempt = state / "attempt-0001"
            authority = authority_for(state)
            observed = {"consumed": False, "claim_absent": False, "endpoint": False}

            def fail_immediately() -> None:
                observed["consumed"] = (
                    state / "authorization-consumed.json"
                ).is_file()
                observed["claim_absent"] = not (
                    state / "execution-claim.json"
                ).exists()
                raise RuntimeError("injected immediately after durable consumption")

            def endpoint_must_not_run(_argv, _environment, _timeout):
                observed["endpoint"] = True
                return {
                    "pid": 0,
                    "exit_code": 99,
                    "stdout": b"",
                    "stderr": b"",
                    "natural_terminal": True,
                }

            self.assertEqual(
                runner.execute_authorized(
                    authority,
                    state=state,
                    attempt=attempt,
                    process_runner=endpoint_must_not_run,
                    post_consumption_hook=fail_immediately,
                ),
                2,
            )
            self.assertTrue(observed["consumed"])
            self.assertTrue(observed["claim_absent"])
            self.assertFalse(observed["endpoint"])
            capture = json.loads(
                (attempt / "endpoint-failure-capture.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                capture["classification"], "RTL_ENDPOINT_EXCEPTION_RuntimeError"
            )
            self.assertEqual(capture["error_type"], "RuntimeError")
            terminal_paths = [
                path
                for path in (
                    state / "terminal-seal.json",
                    state / "fallback/terminal-seal.json",
                )
                if path.is_file()
            ]
            self.assertEqual(len(terminal_paths), 1)
            terminal = json.loads(terminal_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["classification"], "RTL_ENDPOINT_EXCEPTION_RuntimeError"
            )

'''
    return replace_exact(
        text,
        "    def test_success_publishes_endpoint_receipt_agreement_and_terminal(self) -> None:\n",
        test
        + "    def test_success_publishes_endpoint_receipt_agreement_and_terminal(self) -> None:\n",
        "immediate post-consumption fault test",
    )


def successor_constructor() -> str:
    text = BASE_CONSTRUCTOR.read_text(encoding="utf-8").replace(
        OLD_IDENTITY, NEW_IDENTITY
    )
    text = replace_exact(
        text,
        '            "every_runner_controlled_outcome": True,\n',
        '            "every_runner_controlled_outcome": True,\n'
        '            "accountability_start": '
        '"IMMEDIATELY_AFTER_DURABLE_AUTHORIZATION_CONSUMPTION",\n',
        "authority accountability contract",
    )
    text = replace_exact(
        text,
        '                "endpoint_outcome": "FSYNCED_RECEIPT_OR_EXPLICIT_FAILURE_CAPTURE",\n',
        '                "endpoint_outcome": "FSYNCED_RECEIPT_OR_EXPLICIT_FAILURE_CAPTURE",\n'
        '                "post_consumption_accountability": '
        '"IMMEDIATELY_AFTER_DURABLE_AUTHORIZATION_CONSUMPTION",\n',
        "package accountability contract",
    )
    text = replace_exact(
        text,
        '                    "NON_MODEL_RECEIPT_AND_TERMINAL_PUBLICATION_FAILURE_TESTS",\n',
        '                    "NON_MODEL_RECEIPT_AND_TERMINAL_PUBLICATION_FAILURE_TESTS",\n'
        '                    "IMMEDIATE_POST_CONSUMPTION_FAILURE_CAPTURE_AND_TERMINAL_TEST",\n',
        "review requirement",
    )
    return text


def main() -> None:
    constructor_text = successor_constructor()
    with tempfile.TemporaryDirectory(
        prefix=".cf17-0003-sources-", dir=ROOT / "build/ace2_chat_demo"
    ) as temporary:
        source_root = Path(temporary)
        runner_source = source_root / "cf17_terminal_authority_runner.py"
        validator_source = source_root / "cf17_terminal_validate.py"
        test_source = source_root / "cf17_terminal_tests.py"
        runner_source.write_text(successor_runner(), encoding="utf-8")
        validator_source.write_text(successor_validator(), encoding="utf-8")
        test_source.write_text(successor_tests(), encoding="utf-8")

        module = types.ModuleType("_cf17_0003_constructor")
        module.__file__ = str(Path(__file__).resolve())
        exec(compile(constructor_text, module.__file__, "exec"), module.__dict__)
        module.RUNNER_SOURCE = runner_source
        module.VALIDATOR_SOURCE = validator_source
        module.TEST_SOURCE = test_source

        def source_manifest(context: dict[str, Any]) -> dict[str, object]:
            sources = {}
            for path, scope in (
                (module.CONSTRUCTOR, "repository"),
                (BASE_CONSTRUCTOR, "immutable_cf17_0002_constructor_source"),
                (BASE_RUNNER, "immutable_cf17_0002_runner_source"),
                (BASE_VALIDATOR, "immutable_cf17_0002_validator_source"),
                (BASE_TESTS, "immutable_cf17_0002_test_source"),
                (module.MISSION, "external_canonical"),
                (module.CF15 / "bindings.json", "accepted_endpoint_binding"),
                (
                    module.CF16_PACKAGE / "package-manifest.json",
                    "immutable_cf16_package",
                ),
                (module.HOST_EVIDENCE, "immutable_cf16_evidence"),
                (
                    module.CF16_STATE
                    / "attempt-0001/product/rtl-completion/commands.jsonl",
                    "immutable_cf16_partial_endpoint_evidence",
                ),
                (
                    module.CF16_STATE
                    / "attempt-0001/product/rtl-completion/progress.json",
                    "immutable_cf16_partial_endpoint_evidence",
                ),
            ):
                relative = (
                    path.as_posix()
                    if path.is_absolute() and not path.is_relative_to(module.ROOT)
                    else path.relative_to(module.ROOT).as_posix()
                )
                sources[relative] = {"scope": scope, **module.file_record(path)}
            for name, path in context["endpoint_paths"].items():
                relative = (
                    path.as_posix()
                    if not path.is_relative_to(module.ROOT)
                    else path.relative_to(module.ROOT).as_posix()
                )
                sources[relative] = {
                    "scope": f"rtl_endpoint_{name}",
                    **module.file_record(path),
                }
            return {
                "schema": "ace2-r6-cf17-source-constraint-manifest-v1",
                "files": dict(sorted(sources.items())),
                "path_bindings": context["endpoint_path_bindings"],
                "rtl_change": "NONE",
                "constraint_change": "NONE",
                "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
                "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
                "non_sram_area_cap_mm2": 2.0,
                "minimum_frequency_mhz": 100,
            }

        module.source_manifest = source_manifest
        module.main()


if __name__ == "__main__":
    main()
