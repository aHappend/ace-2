#!/usr/bin/env python3

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "evidence" / "verification" / "latest"
LOGS = {
    "icarus_full": ROOT / "evidence" / "frontier" / "latest" / "rtl_shell_sim.log",
    "icarus_smoke": VERIFY / "rtl_shell_smoke_o_proj.log",
    "verilator": VERIFY / "rtl_shell_verilator_oproj.log",
}
HASHED_FILES = [
    "Makefile",
    "verification/tb/ace2_shell_tb.sv",
    "verification/verilator/ace2_shell_oproj_harness.sv",
    "verification/verilator/ace2_shell_oproj_main.cpp",
    "tools/check_shell_throughput_agreement.py",
    "verification/generated/projection_vectors.svh",
    "rtl/ace2_shell.sv",
    "constraints/ace2_rmsnorm_core.sdc",
    "evidence/frontier/latest/rtl_shell_sim.log",
    "evidence/verification/latest/rtl_shell_smoke_o_proj.log",
    "evidence/verification/latest/rtl_shell_verilator_oproj.log",
]
AGGREGATE_RE = re.compile(
    r"^ACE2_SHELL_OPCODE_RESULT opcode=01 vectors=(?P<vectors>\d+) "
    r"writes=(?P<writes>\d+) signature=(?P<signature>[0-9a-fA-F]{16}) "
    r"total_cycles=(?P<total_cycles>\d+) max_cycles=(?P<max_cycles>\d+)$",
    re.MULTILINE,
)
VECTOR_RE = re.compile(
    r"^ACE2_SHELL_VECTOR_RESULT simulator=(?P<simulator>\w+) opcode=01 "
    r"vector=(?P<vector>\d+) writes=(?P<writes>\d+) "
    r"signature=(?P<signature>[0-9a-fA-F]{16}) "
    r"command_accept_cycle=(?P<command_accept_cycle>\d+) "
    r"completion_accept_cycle=(?P<completion_accept_cycle>\d+) "
    r"cycles=(?P<cycles>\d+)$",
    re.MULTILINE,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_log(path: Path) -> tuple[str, dict[str, str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    aggregate_match = AGGREGATE_RE.search(text)
    if aggregate_match is None:
        raise RuntimeError(f"missing aggregate result in {path.relative_to(ROOT)}")

    vectors = [match.groupdict() for match in VECTOR_RE.finditer(text)]
    if not vectors:
        raise RuntimeError(f"missing per-vector results in {path.relative_to(ROOT)}")
    for vector in vectors:
        observed = int(vector["completion_accept_cycle"]) - int(
            vector["command_accept_cycle"]
        )
        if observed != int(vector["cycles"]):
            raise RuntimeError(
                f"invalid handshake interval for vector {vector['vector']} "
                f"in {path.relative_to(ROOT)}"
            )
    return aggregate_match.group(0), aggregate_match.groupdict(), vectors


def comparable_vector(vector: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        vector["vector"],
        vector["writes"],
        vector["signature"].lower(),
        vector["cycles"],
    )


def output_vector(vector: dict[str, str]) -> tuple[str, str, str]:
    return (
        vector["vector"],
        vector["writes"],
        vector["signature"].lower(),
    )


def aggregate_output(aggregate: dict[str, str]) -> tuple[str, str, str]:
    return (
        aggregate["vectors"],
        aggregate["writes"],
        aggregate["signature"].lower(),
    )


def cycle_delta(observed: str, reference: str) -> int:
    return int(observed) - int(reference)


def main() -> None:
    parsed = {name: parse_log(path) for name, path in LOGS.items()}
    full_aggregate = parsed["icarus_full"][1]
    smoke_aggregate = parsed["icarus_smoke"][1]
    verilator_aggregate = parsed["verilator"][1]
    if smoke_aggregate != full_aggregate:
        raise RuntimeError("Icarus full and smoke aggregate results do not agree")
    if aggregate_output(verilator_aggregate) != aggregate_output(full_aggregate):
        raise RuntimeError("aggregate Icarus/Verilator output results do not agree")

    aggregate_cycle_deltas = {
        key: cycle_delta(verilator_aggregate[key], full_aggregate[key])
        for key in ("total_cycles", "max_cycles")
    }
    if any(delta != 0 for delta in aggregate_cycle_deltas.values()):
        raise RuntimeError(
            "aggregate Icarus/Verilator cycle observations do not agree exactly"
        )

    full_vectors = [comparable_vector(vector) for vector in parsed["icarus_full"][2]]
    smoke_vectors = [comparable_vector(vector) for vector in parsed["icarus_smoke"][2]]
    verilator_vectors = [
        comparable_vector(vector) for vector in parsed["verilator"][2]
    ]
    if smoke_vectors != full_vectors:
        raise RuntimeError("Icarus full and smoke per-vector results do not agree")
    if [value[:3] for value in verilator_vectors] != [
        value[:3] for value in full_vectors
    ]:
        raise RuntimeError("per-vector Icarus/Verilator output results do not agree")

    vector_cycle_deltas = [
        cycle_delta(verilator_vectors[index][3], full_vectors[index][3])
        for index in range(len(full_vectors))
    ]
    if any(delta != 0 for delta in vector_cycle_deltas):
        raise RuntimeError(
            "per-vector Icarus/Verilator cycle observations do not agree exactly"
        )

    lines = [
        *(f"{name}={result[0]}" for name, result in parsed.items()),
        "icarus_full_smoke_exact_match=yes",
        "verilator_output_match=yes",
        "cycle_observation_exact_match=yes",
        "verilator_cycle_observation_delta="
        + ",".join(
            f"{key}={value}" for key, value in aggregate_cycle_deltas.items()
        ),
        "command_boundary=cmd_valid&&cmd_ready",
        "completion_boundary=cmd_done_valid&&cmd_done_ready",
        "",
        "per_vector_cycles",
    ]
    for index, expected in enumerate(full_vectors):
        lines.append(
            f"vector={expected[0]} writes={expected[1]} signature={expected[2]} "
            f"icarus_full_cycles={full_vectors[index][3]} "
            f"icarus_smoke_cycles={smoke_vectors[index][3]} "
            f"verilator_cycles={verilator_vectors[index][3]} "
            f"verilator_delta={vector_cycle_deltas[index]}"
        )
    lines.extend(["", "artifact_hashes"])
    lines.extend(
        f"{sha256_file(ROOT / relative)}  {relative}" for relative in HASHED_FILES
    )
    (VERIFY / "shell_throughput_agreement.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("ACE2_SHELL_THROUGHPUT_AGREEMENT_PASS vectors=2 output_match=yes cycle_exact_match=yes")


if __name__ == "__main__":
    main()
