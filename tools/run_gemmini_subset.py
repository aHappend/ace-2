#!/usr/bin/env python3
"""Run the pinned, explicitly incompatible Gemmini transposer subset."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build" / "gemmini-baseline"
SBT_HOME = ROOT / "build" / "gemmini-sbt-home"
PROJECT = ROOT / "benchmark" / "baselines" / "gemmini-transposer"
LOG = ROOT / "benchmark" / "raw" / "latest" / "gemmini_transposer_subset.log"
REVISION = "8c3f9923a44a2fe2c7930587be297d6d4f8c09ca"
IMAGE = "sbtscala/scala-sbt:eclipse-temurin-17.0.13_11_1.10.7_2.13.15"


def run_subset() -> None:
    if not (SOURCE / ".git").exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--quiet",
                "https://github.com/ucb-bar/gemmini.git",
                str(SOURCE),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", REVISION, "--quiet"],
            cwd=SOURCE,
            check=True,
        )

    current_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if current_revision != REVISION:
        raise RuntimeError(f"Gemmini checkout is {current_revision}, expected {REVISION}")

    SBT_HOME.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/sbt-home",
        "-e",
        "GEMMINI_ROOT=/repo/build/gemmini-baseline",
        "-v",
        f"{ROOT}:/repo",
        "-v",
        f"{SBT_HOME}:/sbt-home",
        "-w",
        "/repo/benchmark/baselines/gemmini-transposer",
        IMAGE,
        "sbt",
        "-batch",
        "testOnly gemmini.TransposerUnitTest",
    ]
    with LOG.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Gemmini subset failed with exit status {completed.returncode}; see {LOG}"
        )
    log = LOG.read_text(encoding="utf-8")
    if "Tests: succeeded 2, failed 0" not in log:
        raise RuntimeError("Gemmini subset log does not contain the required pass summary")


def main() -> None:
    run_subset()
    print(f"ACE2_GEMMINI_SUBSET_PASS revision={REVISION} tests=2")


if __name__ == "__main__":
    main()
