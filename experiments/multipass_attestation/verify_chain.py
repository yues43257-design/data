"""Chain verifier + runner for the two-pass attestation probe.

Checks, in order:
  1. cert1 verifies against C1 (bound certificate: circuit/schedule/block hashes);
  2. cert1.input_hash == hash(C0) and cert1.output_hash == hash(C1);
  3. cert2.input_hash == hash(C1) and cert2.output_hash == hash(C2);
  4. chain link: cert1.output_hash == cert2.input_hash.

The verifier deliberately does **not** re-run equivalence between C1 and C2: that
is pass2's claim.  This is exactly the attestation gap the probe measures.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import support_tracker as ST      # noqa: E402
import experiments.multipass_attestation.attacks as A   # noqa: E402
import experiments.multipass_attestation.pipeline as P  # noqa: E402

CHAIN_VERSION = "multipass-chain-v1"


def _global_phase_equal(a, b, tol: float = 1e-9):
    if a.num_clbits or b.num_clbits:
        return None
    ua = np.asarray(Operator(a).data, dtype=complex)
    ub = np.asarray(Operator(b).data, dtype=complex)
    m = ua @ ub.conj().T
    off = float(np.abs(m - np.diag(np.diag(m))).max())
    diagonal = np.diag(m)
    return bool(off < tol and float(np.abs(diagonal - diagonal[0]).max()) < tol)


def verify_chain(entry: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    reasons: List[str] = []
    c0, c1, c2 = entry["c0"], entry["c1"], entry["c2"]
    h0, h1, h2 = (ST.circuit_hash(c0), ST.circuit_hash(c1), ST.circuit_hash(c2))

    try:
        cert1_ok = bool(ST.verify_certificate(
            entry["schedule"], entry["mask"], entry["cert1"],
            circuit=c1, block_ranges=entry["ranges"]))
    except Exception as exc:                      # unsupported gates etc.
        cert1_ok = False
        reasons.append(f"cert1 raised {type(exc).__name__}")
    if not cert1_ok:
        reasons.append("cert1 does not verify against C1")
    if entry["cert1"].get("input_hash") != h0:
        reasons.append("cert1.input_hash != hash(C0)")
    if entry["cert1"].get("output_hash") != h1:
        reasons.append("cert1.output_hash != hash(C1)")
    if entry["cert2"].get("input_hash") != h1:
        reasons.append("cert2.input_hash != hash(C1)")
    if entry["cert2"].get("output_hash") != h2:
        reasons.append("cert2.output_hash != hash(C2)")
    if entry["cert1"].get("output_hash") != entry["cert2"].get("input_hash"):
        reasons.append("chain link broken: cert1.output_hash != cert2.input_hash")

    ok = not reasons
    check_circuit = entry.get("check_circuit", c2)
    intended = entry.get("intended", c2)
    attack_legal = _global_phase_equal(check_circuit, intended)
    return {
        "attack": entry.get("attack", "normal"),
        "note": entry.get("note", "untampered two-pass pipeline"),
        "ok": ok,
        "chain_ok": ok,
        "reasons": reasons,
        "attack_legal": attack_legal,
        "detected": not ok,
        "attack_success": int(ok and attack_legal is False),
        "cert1_bytes": len(json.dumps(entry["cert1"])),
        "cert2_bytes": len(json.dumps(entry["cert2"])),
        "verify_seconds": round(time.perf_counter() - started, 6),
        "pass_times": entry.get("times", {}),
    }


def main() -> int:
    base = P.build_pipeline(0)
    results = [verify_chain(A.attack(name, base)) for name in A.ATTACKS]
    print("=" * 110)
    print(f"  two-pass attestation chain  {CHAIN_VERSION}")
    print("=" * 110)
    print(f"  {'attack':<30}{'legal':>8}{'chainOK':>9}{'detected':>10}"
          f"{'success':>9}{'c1B':>5}{'c2B':>5}{'verify_ms':>11}")
    for r in results:
        print(f"  {r['attack']:<30}{str(r['attack_legal']):>8}"
              f"{str(r['chain_ok']):>9}{str(r['detected']):>10}"
              f"{r['attack_success']:>9}{r['cert1_bytes']:>5}"
              f"{r['cert2_bytes']:>5}{r['verify_seconds'] * 1e3:>11.3f}")
    print("-" * 110)
    wins = [r["attack"] for r in results if r["attack_success"]]
    print(f"  attacks that succeed: {wins}")
    if "a3_compromised_pass2" in wins and not (
            {"a1_pass1_block_substitution", "a2_pass2_substitution", "a4_replay"} & set(wins)):
        verdict = "GO: hash chain detects inconsistency, NOT compromise (A3 forges it)"
    elif "a3_compromised_pass2" not in wins:
        verdict = "A3 also detected: hash chain suffices; second paper needs another threat model"
    else:
        verdict = "all attacks trivially hash-detected: NO-GO"
    print(f"  VERDICT: {verdict}")
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps({
        "version": CHAIN_VERSION,
        "pass_times": base["times"],
        "results": results,
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
