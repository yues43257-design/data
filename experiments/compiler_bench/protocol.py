"""Frozen scoring protocol (v1).  Do not change without bumping the version.

Definitions
  L  legality    : residual is diagonal (relative-phase criterion).
  S  semantics   : emitted circuit equals the intended circuit up to a global phase.
  C  certificate : named properties (see PROPERTIES), never a single boolean.

Rules
  * every verifier answers exactly one question;
  * n/a is an abstention, never a false positive;
  * comparisons use the common decided subset;
  * rates carry Wilson 95% intervals;
  * truth_source: "dense" if N <= DENSE_LIMIT else "qcec" (dense preferred).
"""

from __future__ import annotations

import hashlib
import json

PROTOCOL_VERSION = "compiler-bench-protocol-v1"
DENSE_LIMIT = 8

QUESTIONS = {
    "L": "residual is diagonal (relative-phase legality)",
    "S": "emitted equals intended up to global phase",
    "C": "certificate properties",
}

PROPERTIES = {
    "positive": "a genuine certificate for a legal rewrite must be accepted",
    "illegal_rewrite": "an illegal rewrite must be rejected",
    "binding": "a changed circuit must be rejected",
    "tamper": "a forged/replayed certificate must be rejected",
    "chain": "a two-pass chain must reject a wrong C2",
}

TRUTH_SOURCE = "dense if N <= DENSE_LIMIT else 'qcec' (dense preferred)"


def fingerprint() -> str:
    payload = {"version": PROTOCOL_VERSION, "dense_limit": DENSE_LIMIT,
               "questions": QUESTIONS, "properties": PROPERTIES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
