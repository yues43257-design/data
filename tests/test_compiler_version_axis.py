"""Tests for the compiler-version axis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
AXIS_DIR = ROOT / "experiments" / "compiler_bench"
AXIS_JSON = AXIS_DIR / "version_axis.json"


def test_t_count_handles_zphase_and_toffoli():
    from experiments.compiler_bench.version_axis import t_count_from_gates

    gates = [
        SimpleNamespace(name="T", phase=None),
        SimpleNamespace(name="Tdg", phase=None),
        SimpleNamespace(name="ZPhase", phase=0.25),   # pi/4
        SimpleNamespace(name="ZPhase", phase=1.0),    # 4*pi/4: Clifford
        SimpleNamespace(name="ZPhase", phase=0.75),   # 3*pi/4: one T
        SimpleNamespace(name="Tof", phase=None),       # 7-T synthesis
    ]
    assert t_count_from_gates(gates) == 11


def test_version_axis_worker_covers_all_optimizer_paths():
    proc = subprocess.run(
        [sys.executable, str(AXIS_DIR / "version_axis.py"), "--worker",
         "--circuits", "adder_n3"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["qiskit_version"]
    assert report["pyzx_version"]
    assert len(report["circuits"]) == 1
    outputs = report["circuits"][0]["outputs"]
    for key in ("qiskit_l0", "qiskit_l3", "pyzx_full_reduce", "todd"):
        assert key in outputs, key
        assert "error" not in outputs[key] or outputs[key]["t_count"] is not None


def test_version_axis_artifact_covers_two_environments():
    report = json.loads(AXIS_JSON.read_text(encoding="utf-8"))
    assert report["version"] == "compiler-version-axis-v1"
    assert len(report["environments"]) == 2
    versions = {
        (env["qiskit_version"], env["pyzx_version"])
        for env in report["environments"]
    }
    assert len(versions) == 2
    assert report["summary"]["successful_cells"] == report["summary"]["total_cells"]
    assert report["summary"]["cross_environment_input_mismatches"] == []
    assert report["summary"]["circuits"] >= 8
    for circuit in report["circuits"]:
        assert set(circuit["environments"]) == {"current", "qiskit-1.3"}
        for cell in circuit["environments"].values():
            assert len(cell["input"]["canonical_hash"]) == 64
            for key in ("qiskit_l0", "qiskit_l3", "pyzx_full_reduce", "todd"):
                assert key in cell["outputs"], (circuit["name"], key)
