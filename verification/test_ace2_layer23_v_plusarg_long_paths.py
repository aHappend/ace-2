#!/usr/bin/env python3
"""Regression for long $value$plusargs paths used by the layer-23 V testbench."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTBENCH = ROOT / "verification/tb/ace2_layer23_v_rank1_hybrid_shell_tb.sv"
INPUTS = (
    ("ACTIVATION", "activation", 0x11),
    ("V_WEIGHT", "weight", 0x22),
    ("V_META", "meta", 0x33),
    ("BASELINE", "baseline", 0x44),
    ("CORRECTED", "corrected", 0x55),
    ("RANK", "rank", 0x66),
    ("RANK_S8", "rank_s8", 0x77),
    ("SATURATION", "saturation", 0x88),
)


def fixture_source(path_bits: int, expect_reads: bool) -> str:
    declarations = "\n".join(
        f"  reg [{path_bits - 1}:0] {stem}_path;\n"
        f"  reg [7:0] {stem}_mem [0:0];"
        for _plusarg, stem, _sentinel in INPUTS
    )
    captures = "\n".join(
        f'    if (!$value$plusargs("{plusarg}=%s", {stem}_path)) '
        f'$fatal(1, "missing {plusarg}");\n'
        f'    $display("PATH_{plusarg}=%0s", {stem}_path);'
        for plusarg, stem, _sentinel in INPUTS
    )
    if expect_reads:
        body = "\n".join(
            f"    {stem}_mem[0] = 8'h00;\n"
            f"    $readmemh({stem}_path, {stem}_mem);\n"
            f"    if ({stem}_mem[0] !== 8'h{sentinel:02x}) "
            f'$fatal(1, "{plusarg} read failed");'
            for plusarg, stem, sentinel in INPUTS
        )
        result = '$display("READMEMH_ALL_PASS");'
    else:
        body = "\n".join(
            f"    handle = $fopen({stem}_path, \"r\");\n"
            f"    if (handle != 0) $fatal(1, \"{plusarg} truncated path opened\");"
            for plusarg, stem, _sentinel in INPUTS
        )
        result = '$display("LEGACY_TRUNCATION_REJECT_PASS");'
    return f"""\
module plusarg_long_path_fixture;
{declarations}
  integer handle;
  initial begin
{captures}
{body}
    {result}
    $finish;
  end
endmodule
"""


class PlusargLongPathRegression(unittest.TestCase):
    metrics: dict[str, int] = {}

    def test_all_eight_long_paths_and_legacy_rejection(self) -> None:
        source = TESTBENCH.read_text(encoding="utf-8")
        declarations = re.findall(
            r"reg \[(\d+):0\] "
            r"(activation|weight|meta|baseline|corrected|rank|rank_s8|saturation)_path;",
            source,
        )
        self.assertEqual(
            sorted(declarations),
            sorted((("2047", stem) for _plusarg, stem, _sentinel in INPUTS)),
        )

        with tempfile.TemporaryDirectory(
            prefix="ace2-plusarg-long-path-regression-"
        ) as temporary:
            root = Path(temporary)
            long_directory = root / ("transport-" + "x" * 128)
            long_directory.mkdir()
            paths: dict[str, Path] = {}
            for plusarg, stem, sentinel in INPUTS:
                path = long_directory / f"{stem}-synthetic-input.mem"
                path.write_text(f"{sentinel:02x}\n", encoding="ascii")
                self.assertGreater(len(str(path).encode()), 157)
                self.assertLessEqual(len(str(path).encode()), 256)
                paths[plusarg] = path
            path_lengths = [len(str(path).encode()) for path in paths.values()]

            exact = self.run_fixture(root, 2048, True, paths)
            self.assertIn("READMEMH_ALL_PASS", exact.stdout)
            exact_paths = self.reported_paths(exact.stdout)
            self.assertEqual(exact_paths, {key: str(path) for key, path in paths.items()})

            legacy = self.run_fixture(root, 1024, False, paths)
            self.assertIn("LEGACY_TRUNCATION_REJECT_PASS", legacy.stdout)
            legacy_paths = self.reported_paths(legacy.stdout)
            self.assertEqual(set(legacy_paths), set(paths))
            for plusarg, path in paths.items():
                self.assertNotEqual(legacy_paths[plusarg], str(path))
                self.assertEqual(
                    legacy_paths[plusarg].encode(),
                    str(path).encode()[-128:],
                )
            type(self).metrics = {
                "input_class_count": len(paths),
                "minimum_path_bytes": min(path_lengths),
                "maximum_path_bytes": max(path_lengths),
            }

    def run_fixture(
        self,
        root: Path,
        path_bits: int,
        expect_reads: bool,
        paths: dict[str, Path],
    ) -> subprocess.CompletedProcess[str]:
        fixture = root / f"fixture-{path_bits}.sv"
        binary = root / f"fixture-{path_bits}.vvp"
        fixture.write_text(
            fixture_source(path_bits, expect_reads),
            encoding="ascii",
        )
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(binary), str(fixture)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return subprocess.run(
            [
                "vvp",
                str(binary),
                *(f"+{plusarg}={path}" for plusarg, path in paths.items()),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def reported_paths(stdout: str) -> dict[str, str]:
        return dict(
            line.removeprefix("PATH_").split("=", 1)
            for line in stdout.splitlines()
            if line.startswith("PATH_")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path)
    parser.add_argument("--log", type=Path)
    arguments = parser.parse_args()
    if (arguments.record is None) != (arguments.log is None):
        parser.error("--record and --log must be provided together")
    if arguments.record is not None:
        if arguments.record.exists() or arguments.log.exists():
            parser.error("--record and --log outputs must not already exist")
        arguments.record.parent.mkdir(parents=True, exist_ok=True)
        arguments.log.parent.mkdir(parents=True, exist_ok=True)
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        PlusargLongPathRegression
    )
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    rendered_output = output.getvalue()
    print(rendered_output, end="")
    if not result.wasSuccessful():
        raise SystemExit(1)
    if arguments.record is not None:
        record = {
            "schema_version": 1,
            "status": "PASS_SYNTHETIC_LONG_PLUSARG_PATH_REGRESSION",
            **PlusargLongPathRegression.metrics,
            "configured_buffer_bytes": 256,
            "legacy_buffer_bytes": 128,
            "input_classes": [plusarg for plusarg, _stem, _sentinel in INPUTS],
            "attempt_vectors_used": False,
            "checks": {
                "all_paths_longer_than_157_bytes": True,
                "all_paths_within_256_bytes": True,
                "all_eight_plusargs_retained_exactly": True,
                "all_eight_readmemh_sentinels_loaded": True,
                "legacy_buffers_left_truncated_to_128_bytes": True,
                "legacy_truncated_paths_rejected_by_fopen": True,
            },
        }
        with arguments.record.open("xb") as stream:
            stream.write(
                (
                    json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True)
                    + "\n"
                ).encode("ascii")
            )
        with arguments.log.open("xb") as stream:
            stream.write(rendered_output.encode("ascii"))
