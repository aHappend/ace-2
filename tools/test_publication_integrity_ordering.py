#!/usr/bin/env python3
"""Metadata-only regression for exactly-once marker ordering."""

from __future__ import annotations

import json
import unittest

from check_publication_integrity import (
    verify_exactly_once_runner_ordering,
    verify_runner_ordering_text,
)


class PublicationIntegrityOrderingTest(unittest.TestCase):
    def test_guard_before_marker_is_accepted(self) -> None:
        verify_runner_ordering_text(
            """
            make publication-authorization-preflight
            mkdir "$MARKER_DIR"
            printf '%s\n' consumed > "$MARKER"
            """,
            "safe-fixture",
        )

    def test_guard_after_marker_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "only after authorization marker creation"):
            verify_runner_ordering_text(
                """
                mkdir "$MARKER_DIR"
                printf '%s\n' consumed > "$MARKER"
                make publication-integrity-check
                """,
                "late-fixture",
            )

    def test_missing_guard_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "lacks a publication-integrity pre-marker guard"):
            verify_runner_ordering_text(
                """
                mkdir "$MARKER_DIR"
                printf '%s\n' consumed > "$MARKER"
                """,
                "unguarded-fixture",
            )

    def test_repository_runners_are_guarded_or_hash_locked_and_consumed(self) -> None:
        status = verify_exactly_once_runner_ordering()
        self.assertEqual(status["sealed_historical_runners"], 5)


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PublicationIntegrityOrderingTest)
    result = unittest.TestResult()
    suite.run(result)
    if not result.wasSuccessful():
        details = [f"{case.id()}: {message}" for case, message in result.failures + result.errors]
        raise RuntimeError("; ".join(details))
    print(json.dumps({
        "metadata_only": True,
        "ordering_regressions": result.testsRun,
        "status": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
