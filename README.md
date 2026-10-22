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

**Measured — full QASMBench `small/` scan (42 circuits,
`reports/figure_data/toffoli_recover_scan_all.json`)**

| quantity | value |
|---|---:|
| files scanned | 42 |
| QASM2 parse errors (`swap` etc. not in std) | 7 |
| already contained `ccx` | 4 |
| **new blocks recovered** | **2** (`fredkin_n3`, `toffoli_n3`) |
| **in tracker domain after recovery** | **4** (`fredkin_n3`, `sat_n7`, `simon_n6`, `toffoli_n3`) |
| windows tried | 61,608 |
| recovery time (median / max / total) | **0.016 s / 3.26 s / 13.1 s** |
| timeouts | 0 |

**The five-circuit subset** (`reports/figure_data/toffoli_recover.json`):

| circuit | gates in | blocks recovered | gates out | in tracker domain |
|---|---:|---:|---:|---|
| adder_n4 | 23 | 0 | 23 | no (`no matching pattern`) |
| cat_state_n4 | 4 | 0 | 4 | no |
| **fredkin_n3** | **19** | **1** | **5** | **yes** |
| grover_n2 | 16 | 0 | 16 | no |
| qft_n4 | 13 | 0 | 13 | no |

Every recovered circuit is checked against the measurement-stripped input up to
global phase (max error ≤ 6.7e-16).

`toffoli_scan.py --qasm-dir <dir>` reproduces the scan (per-circuit windows /
seconds / timeout flags).

```bash
.venv/bin/python toffoli_recover.py                       # census + JSON
.venv/bin/python -m pytest tests/test_toffoli_recover.py    # 6 tests
```

## Compiler-version reproducibility

The Qiskit 1.3.0 / pyZX 0.7.0 sensitivity environment is installed outside
`/tmp`:

```bash
experiments/compiler_bench/setup_old_env.sh
export MCX_OLD_PYTHON="${HOME}/venvs/oldver/bin/python"

.venv/bin/python experiments/compiler_bench/version_axis.py
.venv/bin/python qiskit_baseline_versions.py
```

This produces:

- `experiments/compiler_bench/version_axis.json` — 13 circuits × 2 environments
  × 4 compiler paths;
- `reports/figure_data/qiskit_baseline_sensitivity.json` — every Qiskit L3
  claim under the primary 2.5.2 baseline and the 1.3.0 sensitivity baseline.

The paper's primary percentage is **−33.3…−36.8% vs Qiskit 2.5.2 L3**.
Qiskit 1.3.0 gives **−42.9%** on the same legal arithmetic rows; the two must
not be mixed.
