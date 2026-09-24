import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/verify_w4a8_c02_qk_score_rank_margin_repair_design.py"
)
SPEC = importlib.util.spec_from_file_location("repair_design_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class RepairDesignPreflightTest(unittest.TestCase):
    def test_live_task_passes_static_preflight(self) -> None:
        task = preflight.load_json(preflight.TASK_PATH)
        result = preflight.build_result(task)
        self.assertEqual(result["status"], "PASS_REPAIR_DESIGN_STATIC_PREFLIGHT")
        self.assertEqual(
            result["semantics"]["shared_repair_invariant"],
            preflight.EXPECTED_INVARIANT,
        )

    def test_candidate_namespace_is_rejected(self) -> None:
        task = preflight.load_json(preflight.TASK_PATH)
        task["artifact_isolation"]["allowed_output_root"] = (
            "evidence/candidates/forbidden"
        )
        with self.assertRaisesRegex(RuntimeError, "candidate/attempt namespace"):
            preflight.validate_task(task)

    def test_alternative_order_is_frozen(self) -> None:
        task = preflight.load_json(preflight.TASK_PATH)
        task["alternatives"].reverse()
        with self.assertRaisesRegex(RuntimeError, "alternative set or order"):
            preflight.validate_task(task)


if __name__ == "__main__":
    unittest.main()
