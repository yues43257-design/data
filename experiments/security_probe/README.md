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

## Conclusion

**The certificate is not bound to the actual circuit.** `verify_certificate`
re-executes `certify(schedule, mask)` and compares against the *schedule*; if the
adversary keeps the schedule unchanged and emits a different circuit, all three
circuit-level attacks are undetected (`attack_success = 1`).

Detected: unmodelled gates (the tracker's own `UnsupportedGateError`) and
certificate tampering/replay (the mask/support/verdict fields).

## What a fix would need (not implemented here)

* bind the certificate to a **hash of the emitted circuit** (or of the canonical
  gate list) and re-hash at verification time;
* bind each block to a hash of its concrete decomposition;
* verify that the schedule is derived from that same circuit (a translator that
  cannot be skipped).

Without those, the certificate is a statement about a *claimed schedule*, not
about the circuit that will run.

## Reproduce

```bash
.venv/bin/python experiments/security_probe/probe.py
.venv/bin/python -m pytest experiments/security_probe/tests -q   # 8 passed
```
