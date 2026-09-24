import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/diagnose_w4a8_c02_sealed_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("c02_audit", MODULE_PATH)
assert SPEC and SPEC.loader
c02_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c02_audit)


class Scale32AuditTest(unittest.TestCase):
    def test_known_records_decode_exactly(self) -> None:
        self.assertEqual(c02_audit.unpack_scale32(0x00E88000), (0x8000, -24))
        self.assertEqual(c02_audit.scale32_value(0x00008000), 1.0)
        self.assertEqual(c02_audit.scale32_value(0x0000FFFF), 1.999969482421875)

    def test_rejects_reserved_byte_and_denormal(self) -> None:
        with self.assertRaises(ValueError):
            c02_audit.unpack_scale32(0x01008000)
        with self.assertRaises(ValueError):
            c02_audit.unpack_scale32(0x00007FFF)

    def test_result_keeps_later_boundaries_unresolved(self) -> None:
        result = c02_audit.build_result()
        self.assertEqual(result["status"], "PASS_BOUNDED_UNRESOLVED_BOUNDARY")
        self.assertEqual(
            result["boundary_classification"]["earliest_unresolved_boundary"],
            "activation quantization and first projection requantization",
        )
        self.assertFalse(result["prohibitions_observed"]["execute_command_invoked"])


if __name__ == "__main__":
    unittest.main()
