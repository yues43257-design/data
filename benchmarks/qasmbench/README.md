# QASMBench circuits (downloaded)

Source: https://github.com/pnnl/QASMBench (branch `master`), `small/<name>/<name>.qasm`.
Downloaded 2026-10-20 for the domain-boundary census in `qasmbench_census.py`.

These files are **verbatim copies**; see the upstream repository for its license
and citation. They are stored here only so `tail/qasmbench` measurements are
reproducible offline.

Census result (see report §5.57): **0/5 are inside the tracker's declared domain** —
none exposes `ccx` blocks (the Toffolis are pre-decomposed or written as `cu1`),
and all five contain terminal `measure` gates.
