# Certificate-binding security probe (independent, not a paper claim)

**Threat model.** The adversary controls the compiler pass: it emits the circuit
*and* chooses the schedule/certificate. `support_tracker.certify(schedule, mask)`
only looks at block metadata and layer gates; it never hashes the circuit or the
block implementation.

**Base.** N = 5, two downgraded blocks + one CNOT between them (H-free layers),
mask = 3. The genuine certificate is accepted.

| attack | circuit legal | cert accepts | detected | attack success |
|---|---|---|---|---|
| normal | yes | yes | – | 0 |
| **block_substitution** (re-map block 0 `[0,1,2]→[0,2,1]`) | **no** | **yes** | **no** | **1** |
| **nonblock_x** (X on a data qubit) | **no** | **yes** | **no** | **1** |
| **tail_h** (H after the last block) | **no** | **yes** | **no** | **1** |
| gate_smuggle `u3` | no | no (`UnsupportedGateError`) | yes | 0 |
| gate_smuggle `measure` | n/a | no (`UnsupportedGateError`) | yes | 0 |
| cert_forge (`accepted` flipped, support padded) | yes | no | yes | 0 |
| cert_replay (mask 1 cert used for mask 3) | yes | no | yes | 0 |

## Before/after the binding fix

| attack | legal | **unbound** (old API) | **bound** (fixed) | detected |
|---|---|---|---|---|
| normal | yes | accepts | accepts | – |
| block_substitution | no | **accepts** | **rejects** | yes |
| nonblock_x | no | **accepts** | **rejects** | yes |
| tail_h | no | **accepts** | **rejects** | yes |
| gate_smuggle u3 / measure | no | rejects | rejects | yes |
| cert_forge | yes | rejects | rejects | yes |
| cert_replay | yes | rejects | rejects | yes |

**Fix**: `certify(schedule, mask, circuit=..., block_ranges=...)` now adds
`circuit_hash` (canonical gate list), `schedule_hash`, and `block_hashes`
(per-block decomposition); `verify_certificate(..., circuit=..., block_ranges=...)`
recomputes them and compares. The unbound call path is unchanged (backward
compatible) and is shown in the "unbound" column as the pre-fix behaviour.

**Result**: 7/7 non-baseline attacks detected, **0 attack successes**; the
certificate grows from 147 B to 363 B and verification stays sub-millisecond.

## Conclusion

The certificate is **schedule-level** by construction. In a pipeline where the
pass chooses both the circuit and the schedule, it must be **bound to the
circuit hash** (and to each block's decomposition); otherwise a malicious pass
can substitute a block, insert a non-block gate, or append a tail Hadamard while
the unbound certificate still verifies.

## Reproduce

```bash
.venv/bin/python experiments/security_probe/probe.py
.venv/bin/python -m pytest experiments/security_probe/tests -q   # 8 passed
```
