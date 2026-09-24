from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = ROOT / "verification/verilator/ace2_shell_runtime_main.cpp"
PROFILE_TEST_SOURCE = (
    ROOT / "verification/verilator/ace2_runtime_identity_profile_test.cpp"
)
PROFILE_INCLUDE = ROOT / "verification/verilator"


def profile_compile_command() -> list[str]:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise RuntimeError("a C++17 compiler is required")
    return [
        compiler,
        "-std=c++17",
        "-fsyntax-only",
        "-I",
        str(PROFILE_INCLUDE),
        str(PROFILE_TEST_SOURCE),
    ]


class Ace2RuntimeIdentityProfileTest(unittest.TestCase):
    def test_profile_contract_is_compile_time_only(self) -> None:
        completed = subprocess.run(
            profile_compile_command(),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runtime_parser_wires_base_default_and_explicit_opt_in(self) -> None:
        source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "IdentityProfileKind::Base;",
            source,
        )
        self.assertIn('option == "--identity-profile"', source)
        self.assertIn("parse_identity_profile(profile_name)", source)
        self.assertIn("selected_identity_profile == nullptr", source)
        self.assertIn(
            "load_package(arguments.package, *selected_identity_profile)",
            source,
        )

    def test_focused_tests_do_not_invoke_the_runtime_binary(self) -> None:
        command = profile_compile_command()
        self.assertEqual(
            command[0],
            shutil.which(os.environ.get("CXX", "c++")),
        )
        self.assertIn("-fsyntax-only", command)
        self.assertNotIn("-o", command)
        self.assertIn(str(PROFILE_TEST_SOURCE), command)
        self.assertNotIn(str(RUNTIME_SOURCE), command)
        self.assertFalse(any(str(ROOT / "build") in argument for argument in command))


if __name__ == "__main__":
    unittest.main()
