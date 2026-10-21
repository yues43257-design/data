#!/usr/bin/env python3
"""Run the certificate-binding attacks and write results.json.

Usage
-----
    .venv/bin/python experiments/security_probe/probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import experiments.security_probe.attacks as A   # noqa: E402

PROBE_VERSION = "security-probe-v1"


def run(write: bool = True) -> List[Dict[str, Any]]:
    base = A.build_base()
    results = [A.evaluate(base, A.attack(name, base)) for name in A.ATTACKS]
    if write:
        out = Path(__file__).resolve().parent / "results.json"
        out.write_text(json.dumps({"version": PROBE_VERSION,
                                   "circuit": {"N": 5, "blocks": 2, "mask": 3},
                                   "results": results}, indent=2),
                       encoding="utf-8")
    return results


def main() -> int:
    results = run()
    print("=" * 108)
    print(f"  certificate-binding security probe  {PROBE_VERSION}")
    print("=" * 108)
    print(f"  {'attack':<22}{'legal':>8}{'certAcc':>9}{'detected':>10}"
          f"{'success':>9}{'certB':>7}{'verify_s':>10}  error")
    for r in results:
        print(f"  {r['attack']:<22}{str(r['attack_legal']):>8}"
              f"{str(r['certificate_accepts']):>9}{str(r['detected']):>10}"
              f"{r['attack_success']:>9}{r['certificate_bytes']:>7}"
              f"{r['verify_seconds']:>10.6f}  {r['error'] or ''}")
    print("-" * 108)
    wins = [r["attack"] for r in results if r["attack_success"]]
    print(f"  attacks that succeed (certificate accepts an illegal circuit): {wins}")
    print(f"  attacks detected: "
          f"{[r['attack'] for r in results if r['detected']]}")
    out = Path(__file__).resolve().parent / "results.json"
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
