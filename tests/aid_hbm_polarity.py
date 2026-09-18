#!/usr/bin/env python3
"""Host-only check that MPK_AID_POLARITY swaps XCD→AID ranges, not data halves.

Loads aid_hbm.py by path so this does not import mirage/z3/HIP.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "python" / "mirage" / "mpk" / "aid_hbm.py"
_spec = importlib.util.spec_from_file_location("aid_hbm", _SRC)
aid_hbm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(aid_hbm)


def main() -> int:
    os.environ["MPK_AID_POLARITY"] = "0"
    if aid_hbm.polarity() != 0 or aid_hbm.apply_polarity(2, 5) != (2, 5):
        print("FAIL: polarity 0 should keep AID0→XCD0-3", file=sys.stderr)
        return 1
    os.environ["MPK_AID_POLARITY"] = "1"
    if aid_hbm.polarity() != 1 or aid_hbm.apply_polarity(2, 5) != (5, 2):
        print("FAIL: polarity 1 should place XCD0-3 on AID1 range", file=sys.stderr)
        return 1
    os.environ["MPK_AID_POLARITY"] = "2"
    try:
        aid_hbm.polarity()
    except ValueError:
        pass
    else:
        print("FAIL: polarity 2 should raise", file=sys.stderr)
        return 1
    print("AID polarity mapping OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
