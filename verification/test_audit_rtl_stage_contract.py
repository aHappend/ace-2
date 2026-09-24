from __future__ import annotations

import unittest

from tools import audit_rtl_stage_contract as audit


class RtlStageContractAuditTest(unittest.TestCase):
    def test_package_constants_resolve_through_generated_include(self) -> None:
        sources = audit.rtl_include_closure([audit.PACKAGE])
        self.assertEqual(
            [path.relative_to(audit.ROOT).as_posix() for path in sources],
            [
                "rtl/ace2_pkg.sv",
                "rtl/generated/ace2_model_parameters.svh",
            ],
        )
        self.assertEqual(
            audit.parse_package_interface_constants(),
            {
                "ACE2_HIDDEN_SIZE": 896,
                "ACE2_VECTOR_LANES": 16,
                "ACE2_SRAM_BANKS": 8,
                "ACE2_SRAM_ADDR_WIDTH": 12,
            },
        )

    def test_active_input_closure_hash_binds_generated_include(self) -> None:
        records, manifest_sha256 = audit.active_rtl_input_closure()
        paths = {record["path"] for record in records}
        self.assertIn("rtl/generated/ace2_model_parameters.svh", paths)
        self.assertRegex(manifest_sha256, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
