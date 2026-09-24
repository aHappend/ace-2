#!/usr/bin/env python3
"""Protected target boundary reserved by V29; static acceptance never invokes it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def compact_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    args = parser.parse_args()
    raw = args.envelope.read_bytes()
    envelope = json.loads(raw.decode("ascii", "strict"))
    if type(envelope) is not dict or compact_bytes(envelope) != raw:
        return 70
    observed = envelope.get("envelope_sha256")
    unhashed = dict(envelope)
    unhashed.pop("envelope_sha256", None)
    if observed != hashlib.sha256(compact_bytes(unhashed)).hexdigest():
        return 70
    if args.envelope != args.runtime / "authority/internal/frozen-envelope.json":
        return 70
    # V29 is accepted only as a static binding repair. A later, separately
    # authorized additive package must replace this boundary with the protected
    # payload adapter; direct execution therefore fails closed.
    sys.stderr.write("STATIC_ONLY_NO_EXECUTION_AUTHORITY\n")
    return 77


if __name__ == "__main__":
    raise SystemExit(main())
