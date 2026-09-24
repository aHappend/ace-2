#!/usr/bin/env python3
"""Create and seal layer-16 authority package 0017 exclusively."""

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
SEALED_SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0015"
SEALED_SOURCE_REVIEW = ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0015"
FAILED_REFERENCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0016"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0017"
SEALED_SOURCE_ROOT = (
    "a513b7c7003169c39abe39d0c56ec861f3ba59407fcf1a653e6a79ae5e5441fc"
)
ENGINEER_MISSION_ID = "44a2eda3bd5e"
DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
AGENT_IO = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/agent_io.jsonl"
)
SOURCE_MEMBERS = {
    "execute_layer16.py",
    "hostile_controls.py",
    "review_package.py",
    "seal_package.py",
    "validate_nonexecuting.py",
    "validate_review.py",
}
FAILED_REFERENCE_HASHES = {
    "acceptance-contract.json": (
        "99410e785ac7e8b7d56ef14510db83dbbe69d5ecb121882d1084cf0388398c57"
    ),
    "authority.json": (
        "75895dcf2739d00ae0b997d8536ea6bb0da5c6bd7983414cd5e221b4cddb3e3e"
    ),
    "execute_layer16.py": (
        "bdd93b1b3c15f86780cab7e11fe460ae9943aeb565b4acb731396529a658e556"
    ),
    "failed-predecessor-provenance.json": (
        "0ed8aacc6c2ab3a75fde4b9d9cb342ea03af6d8968fd6329df3f47b25c52207d"
    ),
    "hostile_controls.py": (
        "6b1ebc11ea85ef5e2ea0b4115bff113a910d96724512a66a88935815577b1235"
    ),
    "review-request.json": (
        "6693f8291ea0e20e0ca71b7d3ea33ba0b154cc958eda502ab0439e1d0df0e0d8"
    ),
    "review_package.py": (
        "3267ef9990db1a356de16b740daae3a854c46b20a35f4057dbc2c5076805757a"
    ),
    "seal_package.py": (
        "799eaa1f4fc15bb9c1b756108bab8538192f2f9f0b5c8651da4c3e5586d6a2e0"
    ),
    "validate_nonexecuting.py": (
        "d4f752fb5460b6dcaec0fbc27c635eb3d15ba3f0255f0626c906657c23d3976d"
    ),
    "validate_review.py": (
        "f0257cfb12376c459652ba02446a95b57a7148d0942c9cdf30988a742612e918"
    ),
}
SEALED_SOURCE_REVIEW_HASHES = {
    "review-log.json": (
        "0cb1a79ea9c77916b08a58ea0e92b94b1d5bb33fc5e6d65094e066cb65c50eff"
    ),
    "reviewer-adjudication.json": (
        "87cd89561c141310f9b87f1626cdb3448d89a8f31fe565726f670ccbb1e72a17"
    ),
    "reviewer-receipt.json": (
        "2ecbd0d2440ef22c16d6008d50d220f291179af70e7497fa489965d9bd37d216"
    ),
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


def validate_sealed_source() -> None:
    require(
        SEALED_SOURCE.is_dir()
        and not SEALED_SOURCE.is_symlink()
        and stat.S_IMODE(SEALED_SOURCE.stat().st_mode) == 0o555,
        "sealed package0015 directory changed",
    )
    records: dict[str, str] = {}
    for line in (SEALED_SOURCE / "SHA256SUMS").read_text(
        encoding="ascii"
    ).splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "sealed package0015 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(SEALED_SOURCE).as_posix()
        for item in SEALED_SOURCE.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "sealed package0015 member set changed")
    for name, digest in records.items():
        path = SEALED_SOURCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"sealed package0015 member changed: {path}",
        )
    require(
        sha256_file(SEALED_SOURCE / "SHA256SUMS") == SEALED_SOURCE_ROOT
        and (SEALED_SOURCE / "TREE_ROOT.sha256").read_text(
            encoding="ascii"
        ).split()
        == [SEALED_SOURCE_ROOT, "SHA256SUMS"],
        "sealed package0015 root changed",
    )
    require(
        SEALED_SOURCE_REVIEW.is_dir()
        and not SEALED_SOURCE_REVIEW.is_symlink()
        and stat.S_IMODE(SEALED_SOURCE_REVIEW.stat().st_mode) == 0o555
        and {item.name for item in SEALED_SOURCE_REVIEW.iterdir()}
        == set(SEALED_SOURCE_REVIEW_HASHES),
        "sealed package0015 review namespace changed",
    )
    for name, digest in SEALED_SOURCE_REVIEW_HASHES.items():
        path = SEALED_SOURCE_REVIEW / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"sealed package0015 review artifact changed: {path}",
        )


def validate_failed_reference() -> None:
    require(
        FAILED_REFERENCE.is_dir()
        and not FAILED_REFERENCE.is_symlink()
        and stat.S_IMODE(FAILED_REFERENCE.stat().st_mode) == 0o700,
        "failed package0016 directory changed",
    )
    require(
        {item.name for item in FAILED_REFERENCE.iterdir()}
        == set(FAILED_REFERENCE_HASHES),
        "failed package0016 member set or absent sidecars changed",
    )
    for name, digest in FAILED_REFERENCE_HASHES.items():
        path = FAILED_REFERENCE / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"failed package0016 member changed: {path}",
        )


def validate_inputs() -> dict[str, str]:
    validate_sealed_source()
    validate_failed_reference()
    require(DIRECTIVE.is_file() and not DIRECTIVE.is_symlink(), "Manager V6 absent")
    directive = json.loads(DIRECTIVE.read_text(encoding="utf-8"))
    require(
        isinstance(directive, dict)
        and directive.get("revision") == "f1bb5b954b414680ad64d2e9ce1985f0"
        and directive.get("objective_sha256")
        == "43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"
        and isinstance(directive.get("text"), str)
        and "MANAGER GRANT LAYER16 V6" in directive["text"]
        and "package0017" in directive["text"]
        and "joint-record hostile boundary" in directive["text"],
        "active Manager V6 directive changed",
    )
    require(
        sha256_file(DIRECTIVE)
        == "17f4a5bd958abdbb4375a9e8e0a8f5dd6362806aee9c780be0840f5ee3b638f1",
        "active Manager V6 directive bytes changed",
    )
    require(AGENT_IO.is_file() and not AGENT_IO.is_symlink(), "Host agent_io absent")
    for path in (
        TARGET,
        ROOT / "reports/ace2-layer16-runtime-pass-authority-review-0017",
        ROOT / "reports/ace2-layer16-runtime-pass-manager-grant-0017",
        ROOT / "reports/ace2-layer16-continuation-0010-consumption-state-0017",
    ):
        require(not os.path.lexists(path), f"package0017 target preexists: {path}")
    return {
        "revision": directive["revision"],
        "sha256": sha256_file(DIRECTIVE),
        "objective_sha256": directive["objective_sha256"],
    }


FAILED_CONSTRUCTION_CODE = r'''
FAILED_CONSTRUCTION_0016 = (
    ROOT / "reports/ace2-layer16-runtime-pass-authority-0016"
)
FAILED_CONSTRUCTION_0016_HASHES = {
    "acceptance-contract.json": "99410e785ac7e8b7d56ef14510db83dbbe69d5ecb121882d1084cf0388398c57",
    "authority.json": "75895dcf2739d00ae0b997d8536ea6bb0da5c6bd7983414cd5e221b4cddb3e3e",
    "execute_layer16.py": "bdd93b1b3c15f86780cab7e11fe460ae9943aeb565b4acb731396529a658e556",
    "failed-predecessor-provenance.json": "0ed8aacc6c2ab3a75fde4b9d9cb342ea03af6d8968fd6329df3f47b25c52207d",
    "hostile_controls.py": "6b1ebc11ea85ef5e2ea0b4115bff113a910d96724512a66a88935815577b1235",
    "review-request.json": "6693f8291ea0e20e0ca71b7d3ea33ba0b154cc958eda502ab0439e1d0df0e0d8",
    "review_package.py": "3267ef9990db1a356de16b740daae3a854c46b20a35f4057dbc2c5076805757a",
    "seal_package.py": "799eaa1f4fc15bb9c1b756108bab8538192f2f9f0b5c8651da4c3e5586d6a2e0",
    "validate_nonexecuting.py": "d4f752fb5460b6dcaec0fbc27c635eb3d15ba3f0255f0626c906657c23d3976d",
    "validate_review.py": "f0257cfb12376c459652ba02446a95b57a7148d0942c9cdf30988a742612e918",
}
'''


FAILED_CONSTRUCTION_FUNCTIONS = r'''
def validate_failed_construction_0016() -> None:
    require(
        FAILED_CONSTRUCTION_0016.is_dir()
        and not FAILED_CONSTRUCTION_0016.is_symlink()
        and stat.S_IMODE(FAILED_CONSTRUCTION_0016.stat().st_mode) == 0o700,
        "failed package0016 directory changed",
    )
    require(
        {item.name for item in FAILED_CONSTRUCTION_0016.iterdir()}
        == set(FAILED_CONSTRUCTION_0016_HASHES),
        "failed package0016 member set or absent sidecars changed",
    )
    for name, digest in FAILED_CONSTRUCTION_0016_HASHES.items():
        path = FAILED_CONSTRUCTION_0016 / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"failed package0016 member changed: {path}",
        )


def _failed_construction_binding() -> dict[str, Any]:
    return {
        "failure_taxonomy_class": "HOST_JOINT_RECORD_EXCLUSION",
        "package_member_mode": "0444",
        "package_mode": "0700",
        "package_path": FAILED_CONSTRUCTION_0016.relative_to(ROOT).as_posix(),
        "preserved_unchanged": True,
        "seal_sidecars": "ABSENT",
        "member_sha256": FAILED_CONSTRUCTION_0016_HASHES,
    }


'''


HOST_ANCESTRY_FUNCTIONS = r'''
def _is_inert_transport_delta(
    event: dict[str, Any],
    binding: dict[str, str],
) -> bool:
    inner = event["inner"]
    return (
        event.get("host_call_id") == binding["host_call_id"]
        and set(inner) == {"agentId", "data", "id", "timestamp", "type"}
        and inner.get("agentId") == ENGINEER_MISSION_ID
        and inner.get("type") == "host.transport_delta"
        and inner.get("data") == {"transport": "keepalive"}
        and isinstance(inner.get("id"), str)
        and bool(inner["id"])
    )


def _require_exact_host_ancestry(
    events: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    binding: dict[str, str],
) -> None:
    launch, final, final_turn_end, task_completion = selected[:4]
    request_message, read_start, read_completion, parent_turn_end = selected[4:]
    _require_chronology(events, selected)
    require(
        all(
            event.get("host_call_id") == binding["host_call_id"]
            for event in selected
        ),
        "exact Host call id ancestry required",
    )
    require(
        _event_parent_id(final_turn_end) == _event_id(final)
        and _event_parent_id(task_completion) == _event_id(final_turn_end)
        and _event_parent_id(read_start) == _event_id(request_message)
        and _event_parent_id(read_completion) == _event_id(read_start)
        and _event_parent_id(parent_turn_end) == _event_id(read_completion),
        "exact Host parentId ancestry required",
    )
    event_ids = [_event_id(event) for event in events]
    require(
        len(event_ids) == len(set(event_ids)),
        "duplicate Host event ids rejected",
    )
    selected_ids = {_event_id(event) for event in selected}
    window = events[events.index(launch) : events.index(parent_turn_end) + 1]
    for event in window:
        if _event_id(event) in selected_ids:
            continue
        if event["inner"].get("agentId") == ENGINEER_MISSION_ID:
            require(
                _is_inert_transport_delta(event, binding),
                "additional authoritative Engineer-owned Host record rejected",
            )


'''


JOINT_HOSTILE_CONTROLS = r'''    joint = copy.deepcopy(host_events)
    joint.insert(
        2,
        event(
            "assistant.message",
            {
                "phase": "commentary",
                "content": "joint record",
                "turnId": "reviewer-turn",
            },
            event_id="joint-record",
            agent_id=layer16.ENGINEER_MISSION_ID,
            timestamp=2.5,
        ),
    )
    cases.append(expect_rejection("joint-records", lambda: validate_events(joint)))

    authoritative_fields = {
        "reviewer-identity": ("reviewerIdentity", host_agent),
        "final-response": ("content", final_response),
        "hashes": ("resultSha256", "a" * 64),
        "root": ("packageRoot", str(layer16.PACKAGE)),
        "toolCallId": ("toolCallId", task_call),
        "turnId": ("turnId", "reviewer-turn"),
        "call-id": ("callId", "host-call-package0017"),
        "status": ("status", "idle"),
    }
    for field_name, (key, value) in authoritative_fields.items():
        for mode, index, timestamp in (
            ("supply", 1, 1.5),
            ("replace", 1, 2),
            ("endorse", 2, 2.5),
            ("rebind", 3, 3.5),
        ):
            candidate = copy.deepcopy(host_events)
            authoritative_value = "rebound" if mode == "rebind" else value
            record = event(
                "assistant.message",
                {key: authoritative_value},
                event_id=f"joint-{mode}-{field_name}",
                agent_id=layer16.ENGINEER_MISSION_ID,
                timestamp=timestamp,
            )
            if field_name == "call-id" and mode == "rebind":
                record["host_call_id"] = "host-call-rebound"
            if mode == "replace":
                candidate[index] = record
            else:
                candidate.insert(index, record)
            cases.append(
                expect_rejection(
                    f"joint-{mode}-{field_name}",
                    lambda candidate=candidate: validate_events(candidate),
                )
            )

    inert = copy.deepcopy(host_events)
    inert.insert(
        1,
        event(
            "host.transport_delta",
            {"transport": "keepalive"},
            event_id="inert-transport-delta",
            agent_id=layer16.ENGINEER_MISSION_ID,
            timestamp=1.5,
        ),
    )
    validate_events(inert)
    cases.append(
        {
            "name": "inert-transport-delta",
            "result": "PASS_ALLOWED_WITHOUT_AUTHORITATIVE_FIELDS",
        }
    )

    wrong_final_parent = copy.deepcopy(host_events)
    wrong_final_parent[2]["inner"]["parentId"] = "wrong-final-parent"
    cases.append(
        expect_rejection(
            "wrong-final-turn-parent",
            lambda: validate_events(wrong_final_parent),
        )
    )
    wrong_task_parent = copy.deepcopy(host_events)
    wrong_task_parent[3]["inner"]["parentId"] = "wrong-task-parent"
    cases.append(
        expect_rejection(
            "wrong-task-completion-parent",
            lambda: validate_events(wrong_task_parent),
        )
    )
'''


def common_transform(text: str) -> str:
    text = text.replace("0016", "0017")
    text = text.replace("860efb205a06", ENGINEER_MISSION_ID)
    text = text.replace("_v9", "_v10")
    text = text.replace("-v9", "-v10")
    return text


def transform_executor(text: str, directive: dict[str, str]) -> str:
    text = common_transform(text)
    text = replace_once(
        text,
        'ACTIVE_DIRECTIVE_REVISION = "c56c92b1581f45119a329bb9569c90ef"',
        f'ACTIVE_DIRECTIVE_REVISION = "{directive["revision"]}"',
    )
    text = replace_once(
        text,
        '"433f42caa894d1ddf8af6ff16995f229705c877d4c02d8e235449caf27bac3be"',
        f'"{directive["sha256"]}"',
    )
    duplicate_0014 = '''    "0014": (
        ROOT / "reports/ace2-layer16-runtime-pass-authority-0014",
        "d8c0f7ebf53acdbfe46a94678502d8be3ac95d3019be7afb47231ce80a36b67b",
        "HOST_TASK_RESULT_RAW_SERIALIZATION_SUBSTRING",
    ),
'''
    require(text.count(duplicate_0014) == 2, "package0016 duplicate lineage changed")
    first = text.index(duplicate_0014)
    second = text.index(duplicate_0014, first + len(duplicate_0014))
    text = text[:second] + text[second + len(duplicate_0014) :]
    text = replace_once(
        text,
        "\nSTALE_REVIEWER_IDENTITIES = {",
        FAILED_CONSTRUCTION_CODE + "\nSTALE_REVIEWER_IDENTITIES = {",
    )
    text = replace_once(
        text,
        "def _failed_lifecycle_binding() -> dict[str, Any]:",
        FAILED_CONSTRUCTION_FUNCTIONS
        + "def _failed_lifecycle_binding() -> dict[str, Any]:",
    )
    text = replace_once(
        text,
        '''        "regression": (
            "HOST_PARENT_ID_ANCESTRY_AND_EXACT_FINAL_TURN"
        ),''',
        '''        "regression": (
            "HOST_PARENT_ID_STRUCTURED_RESULT_AND_JOINT_RECORD_EXCLUSION"
        ),
        "unsealed_failed_construction": _failed_construction_binding(),''',
    )
    text = replace_once(
        text,
        "def validate_host_completion_events(",
        HOST_ANCESTRY_FUNCTIONS + "def validate_host_completion_events(",
    )
    text = replace_once(
        text,
        '''    _require_exact_final(
        read_final_response,
        final_response,
        final_response_sha256,
        "read_agent",
    )
    return {''',
        '''    _require_exact_final(
        read_final_response,
        final_response,
        final_response_sha256,
        "read_agent",
    )
    _require_exact_host_ancestry(
        events,
        [
            launch,
            final,
            final_turn_end,
            task_completion,
            request_message,
            read_start,
            read_completion,
            parent_turn_end,
        ],
        binding,
    )
    return {''',
    )
    text = replace_once(
        text,
        '''def _validate_bound_evidence(*, current_directive_required: bool) -> None:
    for path, tree_root, _failure_class in PRESERVED_FAILED_PACKAGES.values():''',
        '''def _validate_bound_evidence(*, current_directive_required: bool) -> None:
    validate_failed_construction_0016()
    for path, tree_root, _failure_class in PRESERVED_FAILED_PACKAGES.values():''',
    )
    text = replace_once(
        text,
        '''            MANAGER_DIRECTIVE,
            CHECKPOINT,
            *(''',
        '''            MANAGER_DIRECTIVE,
            CHECKPOINT,
            *(
                FAILED_CONSTRUCTION_0016 / name
                for name in sorted(FAILED_CONSTRUCTION_0016_HASHES)
            ),
            *(''',
    )
    text = text.replace(
        '"failed_lifecycle_packages": "PASS_0009_0011_0012_0013_0014_0015_PRESERVED",',
        '"failed_lifecycle_packages": "PASS_0009_0011_0012_0013_0014_0015_PRESERVED",\n'
        '        "failed_package0016": "PASS_UNSEALED_EXACTLY_PRESERVED",',
    )
    return text


def transform_hostile(text: str) -> str:
    text = common_transform(text)
    text = replace_once(
        text,
        '''            event_id="reviewer-turn-end",
            agent_id=host_agent,
            timestamp=3,''',
        '''            event_id="reviewer-turn-end",
            parent_id="reviewer-final",
            agent_id=host_agent,
            timestamp=3,''',
    )
    text = replace_once(
        text,
        '''            event_id="task-complete",
            parent_id="task-start",
            agent_id=host_agent,''',
        '''            event_id="task-complete",
            parent_id="reviewer-turn-end",
            agent_id=host_agent,''',
    )
    start = text.index("    joint = copy.deepcopy(host_events)")
    end = text.index('    for state in ("running", "cancelled", "failed"):', start)
    text = text[:start] + JOINT_HOSTILE_CONTROLS + text[end:]
    return text


def transformed_sources(directive: dict[str, str]) -> dict[str, str]:
    transforms = {
        "execute_layer16.py": lambda value: transform_executor(value, directive),
        "hostile_controls.py": transform_hostile,
        "review_package.py": common_transform,
        "seal_package.py": common_transform,
        "validate_nonexecuting.py": common_transform,
        "validate_review.py": common_transform,
    }
    result = {
        name: transforms[name](
            (FAILED_REFERENCE / name).read_text(encoding="utf-8")
        )
        for name in sorted(SOURCE_MEMBERS)
    }
    for name, text in result.items():
        compile(text, str(TARGET / name), "exec")
        require("0016-reviewer" not in text, f"stale review task in {name}")
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
        "package0017 seal failed: " + result.stderr.decode("utf-8", "replace"),
    )
    validate_sealed_source()
    validate_failed_reference()
    sys.stdout.buffer.write(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
