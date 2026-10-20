# MCX — relative-phase rewrites with a checkable certificate

(The project report lives in `reports/strategy_b_experiments.md`; the paper draft
in `reports/paper_draft.md`.)

## Toffoli recovery (`toffoli_recover.py`)

The support tracker works on **Toffoli/MCX blocks**. Gate-level files usually
ship the 7-T decomposition instead, so there is nothing to certify. This
standalone preprocessor rewrites gate-level circuits back into blocks.

**Supported**

- contiguous windows of at most 20 gates acting on exactly **3 qubits** whose
  unitary equals a Toffoli **up to global phase** (`U = e^{iφ}·ccx`, tolerance
  `1e-9`); the target is detected automatically (all three choices are tried);
- the Qiskit 7-T decomposition and common variants — either through the fast-path
  label or the generic window search (the report says which one matched);
- optional **terminal** measurement stripping (`strip_terminal_measurements=True`,
  the default). Every other gate is copied unchanged.

**Not supported (on purpose)**

- **relative-phase** variants (RCCX): they differ by a *diagonal* operator, not
  a global phase, so they are **never** matched — erasing them would change the
  semantics;
- **mid-circuit** measurement, `reset`, or classical control: reported, never
  removed;
- windows interleaved with gates on other qubits (the window must act on exactly
  three);
- re-synthesis front-ends: a decomposition that is not exactly a Toffoli stays
  as-is. `benchmarks/qasmbench/adder_n4.qasm` is such a case (8 T total, no
  exact Toffoli window) and the report says `why: no matching pattern`.

**Measured (QASMBench, `reports/figure_data/toffoli_recover.json`)**

| circuit | gates in | blocks recovered | gates out | in tracker domain |
|---|---:|---:|---:|---|
| adder_n4 | 23 | 0 | 23 | no (`no matching pattern`) |
| cat_state_n4 | 4 | 0 | 4 | no |
| **fredkin_n3** | **19** | **1** | **5** | **yes** |
| grover_n2 | 16 | 0 | 16 | no |
| qft_n4 | 13 | 0 | 13 | no |

Every recovered circuit is checked against the measurement-stripped input up to
global phase (max error ≤ 6.7e-16).

```bash
.venv/bin/python toffoli_recover.py                       # census + JSON
.venv/bin/python -m pytest tests/test_toffoli_recover.py    # 6 tests
```
