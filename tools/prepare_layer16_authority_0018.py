#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0018 exclusively."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path("/home/argustest/ace-2")
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0017"
SOURCE_REVIEW = ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0017"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0018"
DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
SOURCE_ROOT = "09aefc03902c4ea11378e2b9be3b948d2573d73a27c268c2a3e97b38294df2df"
ENGINEER_MISSION_ID = "f1a82b0e164e"
SOURCE_REVIEW_HASHES = {
    "review-log.json": (
        "79d1104b4dbb37c5b67b3cafe41a9b34e0416f695a909dd59ec5b05cdd62dd7d"
    ),
    "reviewer-adjudication.json": (
        "b35e4d8e2263f537029574444981e7d97316b46f7d6b5f6875d73500aeb63657"
    ),
    "reviewer-receipt.json": (
        "3206cf52196695a4c6666245bd81d40222d552bcb720771d1d2fce01dd2dab58"
    ),
}
SOURCE_MEMBERS = {
    "execute_layer16.py",
    "hostile_controls.py",
    "review_package.py",
    "seal_package.py",
    "validate_nonexecuting.py",
    "validate_review.py",
}


class PreparationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, value: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_once(text: str, old: str, new: str) -> str:
    require(text.count(old) == 1, f"source fragment count changed: {old[:80]!r}")
    return text.replace(old, new)


def validate_source() -> None:
    require(
        SOURCE.is_dir()
        and not SOURCE.is_symlink()
        and stat.S_IMODE(SOURCE.stat().st_mode) == 0o555,
        "sealed package0017 directory changed",
    )
    records: dict[str, str] = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "sealed package0017 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SOURCE).as_posix()
        for item in SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "sealed package0017 member set changed")
    for name, digest in records.items():
        path = SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"sealed package0017 member changed: {path}",
        )
    require(
        sha256_file(SOURCE / "SHA256SUMS") == SOURCE_ROOT
        and (SOURCE / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
        == [SOURCE_ROOT, "SHA256SUMS"],
        "sealed package0017 root changed",
    )
    require(
        SOURCE_REVIEW.is_dir()
        and not SOURCE_REVIEW.is_symlink()
        and stat.S_IMODE(SOURCE_REVIEW.stat().st_mode) == 0o555
        and {item.name for item in SOURCE_REVIEW.iterdir()}
        == set(SOURCE_REVIEW_HASHES),
        "sealed package0017 PASS namespace changed",
    )
    for name, digest in SOURCE_REVIEW_HASHES.items():
        path = SOURCE_REVIEW / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"sealed package0017 PASS artifact changed: {path}",
        )


def validate_inputs() -> dict[str, str]:
    validate_source()
    require(DIRECTIVE.is_file() and not DIRECTIVE.is_symlink(), "Manager V7 absent")
    directive = json.loads(DIRECTIVE.read_text(encoding="utf-8"))
    require(
        isinstance(directive, dict)
        and directive.get("revision") == "16f8501da0d748e3bac140c6088bac46"
        and directive.get("objective_sha256")
        == "43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"
        and isinstance(directive.get("text"), str)
        and "MANAGER GRANT LAYER16 V7" in directive["text"]
        and "package0018" in directive["text"]
        and all(
            marker in directive["text"]
            for marker in ("read_agent", "model", "optional", "metadata")
        ),
        "active Manager V7 directive changed",
    )
    require(
        sha256_file(DIRECTIVE)
        == "3f0a1a997a88c38c210b57fa3fe121d097f640ec387d57e5b83e7a8eaf8c350d",
        "active Manager V7 directive bytes changed",
    )
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0018",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0018",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0018",
    ):
        require(not os.path.lexists(path), f"package0018 target preexists: {path}")
    return {
        "revision": directive["revision"],
        "sha256": sha256_file(DIRECTIVE),
        "objective_sha256": directive["objective_sha256"],
    }


FRAMING_FUNCTIONS = r'''def _structured_result(
    event: dict[str, Any],
    label: str,
) -> dict[str, str]:
    data = _event_data(event)
    require("result" in data, f"{label} result field required")
    result = data["result"]
    require(
        isinstance(result, dict)
        and set(result) == {"content", "detailedContent"}
        and isinstance(result["content"], str)
        and isinstance(result["detailedContent"], str),
        f"{label} result must use the exact structured Host schema",
    )
    return result


def _require_exact_final(
    observed: str,
    expected: str,
    expected_sha256: str,
    label: str,
) -> None:
    require(observed == expected, f"{label} final-response content mismatch")
    require(
        hashlib.sha256(observed.encode("utf-8")).hexdigest() == expected_sha256,
        f"{label} final-response hash mismatch",
    )


def _read_agent_final_response(
    result: dict[str, str],
    expected_agent_id: str,
    supported_task_state: str,
    expected_payload: dict[str, str] | None,
) -> str:
    marker = "\n\n[Turn 0]\n"
    require(
        result["content"].count(marker) == 1,
        "exactly one read_agent Turn 0 delimiter required",
    )
    header, response = result["content"].split(marker)
    require(
        "\n" not in header
        and re.search(r"(?m)^\[Turn [0-9]+\]$", response) is None,
        "extra read_agent turns are rejected",
    )
    match = re.fullmatch(
        r"Agent is idle \(waiting for messages\)\. "
        r"agent_id: (?P<agent_id>[^,\n]+), "
        r"agent_type: (?P<agent_type>[^,\n]+), "
        r"status: (?P<status>[^,\n]+), "
        r"description: (?P<description>[^,\n]+), "
        r"elapsed: (?P<elapsed>[0-9]+s), "
        r"total_turns: (?P<turns>[0-9]+)"
        r"(?:, model: (?P<model>[^,\n]+))?",
        header,
    )
    require(match is not None, "read_agent result framing is malformed")
    require(
        supported_task_state == "idle"
        and match.group("agent_id") == expected_agent_id
        and match.group("agent_type") == "general-purpose"
        and match.group("status") == "idle"
        and bool(match.group("description"))
        and match.group("turns") == "1",
        "read_agent result identity, state, or turn count mismatch",
    )
    require(
        result["detailedContent"] == f"<agent {expected_agent_id} idle>",
        "read_agent detailed result identity or state mismatch",
    )
    try:
        payload = _parse_json_no_duplicates(response)
    except json.JSONDecodeError as error:
        raise Layer16Error("read_agent Turn 0 response must be exact JSON") from error
    required = {
        "adjudication_sha256",
        "agent_id",
        "package_tree_root_sha256",
        "receipt_sha256",
        "review_log_sha256",
        "status",
    }
    require(
        isinstance(payload, dict)
        and set(payload) == required
        and all(isinstance(payload[key], str) for key in required)
        and payload["agent_id"] == expected_agent_id
        and payload["status"]
        == "PASS_ARTIFACTS_CREATED_AWAITING_SUPPORTED_TASK_RETURN"
        and all(
            re.fullmatch(r"[0-9a-f]{64}", payload[key]) is not None
            for key in (
                "adjudication_sha256",
                "package_tree_root_sha256",
                "receipt_sha256",
                "review_log_sha256",
            )
        ),
        "read_agent JSON identity, root, hashes, or status mismatch",
    )
    if expected_payload is not None:
        require(
            payload == expected_payload,
            "read_agent JSON does not exactly bind reviewed artifacts",
        )
    return response


'''


def common_transform(text: str) -> str:
    text = text.replace("0017", "0018")
    text = text.replace("44a2eda3bd5e", ENGINEER_MISSION_ID)
    text = text.replace("_v10", "_v11")
    text = text.replace("-v10", "-v11")
    return text


def transform_executor(text: str, directive: dict[str, str]) -> str:
    text = common_transform(text)
    text = replace_once(
        text,
        'ACTIVE_DIRECTIVE_REVISION = "f1bb5b954b414680ad64d2e9ce1985f0"',
        f'ACTIVE_DIRECTIVE_REVISION = "{directive["revision"]}"',
    )
    text = replace_once(
        text,
        '"17f4a5bd958abdbb4375a9e8e0a8f5dd6362806aee9c780be0840f5ee3b638f1"',
        f'"{directive["sha256"]}"',
    )
    predecessor = '''    "0015": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0015",
        "a513b7c7003169c39abe39d0c56ec861f3ba59407fcf1a653e6a79ae5e5441fc",
        "HOST_ASSISTANT_TURN_END_PARENT_TOOL_CALL_ID_SCHEMA",
    ),
'''
    text = replace_once(
        text,
        predecessor,
        predecessor
        + '''    "0017": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0017",
        "09aefc03902c4ea11378e2b9be3b948d2573d73a27c268c2a3e97b38294df2df",
        "HOST_READ_AGENT_REQUIRED_MODEL_FIELD",
    ),
''',
    )
    start = text.index("def _structured_result(", text.index("def _task_launch("))
    end = text.index("def _is_inert_transport_delta(", start)
    duplicate_region = text[start:end]
    require(
        duplicate_region.count("def _structured_result(") == 2
        and duplicate_region.count("def _require_exact_final(") == 2
        and duplicate_region.count("def _read_agent_final_response(") == 2,
        "package0017 shadowed framing definitions changed",
    )
    text = text[:start] + FRAMING_FUNCTIONS + text[end:]
    text = replace_once(
        text,
        '''def validate_host_completion_events(
    events: list[dict[str, Any]],
    expected_agent_id: str,
    supported_task_state: str,
) -> dict[str, str]:''',
        '''def validate_host_completion_events(
    events: list[dict[str, Any]],
    expected_agent_id: str,
    supported_task_state: str,
    expected_read_agent_payload: dict[str, str] | None = None,
) -> dict[str, str]:''',
    )
    text = replace_once(
        text,
        '''    read_final_response = _read_agent_final_response(
        read_result,
        expected_agent_id,
        supported_task_state,
    )''',
        '''    read_final_response = _read_agent_final_response(
        read_result,
        expected_agent_id,
        supported_task_state,
        expected_read_agent_payload,
    )''',
    )
    text = replace_once(
        text,
        '''    host = validate_host_completion_events(
        _host_stream_events(),
        expected_agent_id,
        supported_task_state,
    )''',
        '''    expected_payload = {
        "adjudication_sha256": sha256_file(REVIEW),
        "agent_id": expected_agent_id,
        "package_tree_root_sha256": package_root,
        "receipt_sha256": sha256_file(REVIEW_RECEIPT),
        "review_log_sha256": sha256_file(REVIEW_LOG),
        "status": "PASS_ARTIFACTS_CREATED_AWAITING_SUPPORTED_TASK_RETURN",
    }
    host = validate_host_completion_events(
        _host_stream_events(),
        expected_agent_id,
        supported_task_state,
        expected_payload,
    )''',
    )
    text = text.replace(
        '"HOST_PARENT_ID_STRUCTURED_RESULT_AND_JOINT_RECORD_EXCLUSION"',
        '"HOST_PARENT_ID_JOINT_RECORD_AND_OPTIONAL_MODEL_FRAMING"',
    )
    return text


def transform_hostile(text: str) -> str:
    text = common_transform(text)
    text = replace_once(
        text,
        '    host_agent = "task-agent-reviewer-0018"\n'
        '    task_call = "call_layer16_package0018"\n'
        '    read_call = "call_read_agent_package0018"\n'
        '    final_response = "PACKAGE0018 REVIEW PASS\\nstructured Host result"\n',
        '''    host_agent = "00000000-0000-4000-8000-000000001018"
    task_call = "call_layer16_package0018"
    read_call = "call_read_agent_package0018"
    expected_payload = {
        "adjudication_sha256": "c" * 64,
        "agent_id": host_agent,
        "package_tree_root_sha256": "a" * 64,
        "receipt_sha256": "e" * 64,
        "review_log_sha256": "d" * 64,
        "status": "PASS_ARTIFACTS_CREATED_AWAITING_SUPPORTED_TASK_RETURN",
    }
    final_response = layer16.canonical_bytes(expected_payload).decode("ascii").rstrip("\\n")
''',
    )
    start = text.index("    read_content = (")
    end = text.index("    host_events = [", start)
    text = (
        text[:start]
        + '''    read_header = (
        "Agent is idle (waiting for messages). "
        f"agent_id: {host_agent}, agent_type: general-purpose, status: idle, "
        "description: Adjudicate sealed package0018, elapsed: 1s, total_turns: 1"
    )
    read_content_without_model = read_header + "\\n\\n[Turn 0]\\n" + final_response
    read_content_with_model = (
        read_header + ", model: reviewer-model\\n\\n[Turn 0]\\n" + final_response
    )
    read_content = read_content_without_model
'''
        + text[end:]
    )
    text = replace_once(
        text,
        '    layer16.validate_host_completion_events(host_events, host_agent, "idle")\n',
        '''    layer16.validate_host_completion_events(
        host_events, host_agent, "idle", expected_payload
    )
    with_model = copy.deepcopy(host_events)
    with_model[6]["inner"]["data"]["result"]["content"] = read_content_with_model
    layer16.validate_host_completion_events(
        with_model, host_agent, "idle", expected_payload
    )
''',
    )
    text = replace_once(
        text,
        '        layer16.validate_host_completion_events(value, agent_id, "idle")\n',
        '''        layer16.validate_host_completion_events(
            value, agent_id, "idle", expected_payload
        )
''',
    )
    attack_start = text.index("    mismatched_agent = copy.deepcopy(host_events)")
    attack_end = text.index("    missing_final = copy.deepcopy(host_events)", attack_start)
    framing_attacks = r'''    def reject_framing_attack_set() -> None:
        attacks: list[list[dict[str, object]]] = []
        wrong_agent = copy.deepcopy(host_events)
        wrong_agent[6]["inner"]["data"]["result"]["content"] = (
            read_content_without_model.replace(host_agent, "task-agent-forged-0018")
        )
        attacks.append(wrong_agent)
        fragments = {
            "agent_id": f"agent_id: {host_agent}, ",
            "agent_type": "agent_type: general-purpose, ",
            "status": "status: idle, ",
            "description": "description: Adjudicate sealed package0018, ",
            "elapsed": "elapsed: 1s, ",
            "total_turns": "total_turns: 1",
        }
        for fragment in fragments.values():
            missing_field = copy.deepcopy(host_events)
            missing_field[6]["inner"]["data"]["result"]["content"] = (
                read_content_without_model.replace(fragment, "", 1)
            )
            attacks.append(missing_field)
            duplicate_field = copy.deepcopy(host_events)
            duplicate_field[6]["inner"]["data"]["result"]["content"] = (
                read_content_without_model.replace(fragment, fragment + fragment, 1)
            )
            attacks.append(duplicate_field)
        extra_turn = copy.deepcopy(host_events)
        extra_turn[6]["inner"]["data"]["result"]["content"] = (
            read_content_without_model + "\n[Turn 1]\n{}"
        )
        attacks.append(extra_turn)
        duplicate_turn = copy.deepcopy(host_events)
        duplicate_turn[6]["inner"]["data"]["result"]["content"] = (
            read_content_without_model.replace(
                "\n\n[Turn 0]\n", "\n\n[Turn 0]\n[Turn 0]\n", 1
            )
        )
        attacks.append(duplicate_turn)
        conflicting_model = copy.deepcopy(host_events)
        conflicting_model[6]["inner"]["data"]["result"]["content"] = (
            read_content_with_model.replace(
                ", model: reviewer-model",
                ", model: reviewer-model, model: conflicting-model",
            )
        )
        attacks.append(conflicting_model)
        for key, value in (
            ("agent_id", "00000000-0000-4000-8000-000000009999"),
            ("package_tree_root_sha256", "9" * 64),
            ("adjudication_sha256", "8" * 64),
            ("status", "PASS"),
        ):
            payload = copy.deepcopy(expected_payload)
            payload[key] = value
            wrong_payload = copy.deepcopy(host_events)
            wrong_payload[6]["inner"]["data"]["result"]["content"] = (
                read_header
                + "\n\n[Turn 0]\n"
                + layer16.canonical_bytes(payload).decode("ascii").rstrip("\n")
            )
            attacks.append(wrong_payload)
        for attack in attacks:
            try:
                validate_events(attack)
            except layer16.Layer16Error:
                continue
            raise RuntimeError("read_agent framing attack unexpectedly accepted")
        raise layer16.Layer16Error("all read_agent framing attacks rejected")

    cases.append(
        expect_rejection("mismatched-result-agent", reject_framing_attack_set)
    )
'''
    text = text[:attack_start] + framing_attacks + text[attack_end:]
    text = replace_once(
        text,
        '''        "cases": cases,
        "execution_performed": False,''',
        '''        "cases": cases,
        "execution_performed": False,
        "framing_fixtures": {
            "with_model_sha256": layer16.hashlib.sha256(
                read_content_with_model.encode("utf-8")
            ).hexdigest(),
            "without_model_sha256": layer16.hashlib.sha256(
                read_content_without_model.encode("utf-8")
            ).hexdigest(),
        },''',
    )
    return text


def transform_seal(text: str) -> str:
    text = common_transform(text)
    return replace_once(
        text,
        '''    controls = _load(
        "hostile_controls_layer16_seal",
        PACKAGE / "hostile_controls.py",
    ).run_controls()
    write_json(PACKAGE / "hostile-controls.json", controls)''',
        '''    controls = _load(
        "hostile_controls_layer16_seal",
        PACKAGE / "hostile_controls.py",
    ).run_controls()
    layer16.require(
        len(controls["cases"]) == 86,
        "exactly 86 retained hostile controls required",
    )
    write_json(PACKAGE / "hostile-controls.json", controls)''',
    )


def transformed_sources(directive: dict[str, str]) -> dict[str, str]:
    transforms = {
        "execute_layer16.py": lambda value: transform_executor(value, directive),
        "hostile_controls.py": transform_hostile,
        "review_package.py": common_transform,
        "seal_package.py": transform_seal,
        "validate_nonexecuting.py": common_transform,
        "validate_review.py": common_transform,
    }
    result = {
        name: transforms[name]((SOURCE / name).read_text(encoding="utf-8"))
        for name in sorted(SOURCE_MEMBERS)
    }
    for name, text in result.items():
        compile(text, str(TARGET / name), "exec")
        require("0017-reviewer" not in text, f"stale review task in {name}")
    return result


def main() -> int:
    directive = validate_inputs()
    sources = transformed_sources(directive)
    os.mkdir(TARGET, mode=0o700)
    for name, text in sources.items():
        write_exclusive(TARGET / name, text.encode("utf-8"))
    result = subprocess.run(
        [
            "/home/argustest/miniconda3/bin/python3.13",
            "-B",
            str(TARGET / "seal_package.py"),
        ],
        cwd=ROOT,
        env={
            "LC_ALL": "C",
            "PATH": "/home/argustest/miniconda3/bin:/usr/bin:/bin",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONSAFEPATH": "1",
            "PYTHONUTF8": "1",
        },
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        result.returncode == 0,
        "package0018 seal failed: " + result.stderr.decode("utf-8", "replace"),
    )
    validate_source()
    sys.stdout.buffer.write(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
