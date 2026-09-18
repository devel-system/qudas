"""WP-0 / WP-1: core export, QuData facade, deprecation contracts."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from qudas import QuData
from qudas.annealing import QdAnnealingIR
from qudas.core import (
    QdExecBase,
    QdExecutorBase,
    QdInBase,
    QdInputBase,
    QdOutBase,
    QdOutputBase,
)


def test_package_version_is_not_placeholder():
    import qudas

    assert qudas.__version__ == "0.2.6"


def test_qudata_input_returns_annealing_ir():
    qubo = {("q0", "q1"): 1.0, ("q2", "q2"): -1.0}
    ir = QuData.input(qubo)
    assert isinstance(ir, QdAnnealingIR)
    assert ir.prob == qubo
    assert ir.qubo == qubo


def test_qudata_input_mode_gate_not_supported():
    with pytest.raises(NotImplementedError, match="qudas.gate"):
        QuData.input(mode="gate")


def test_qudata_input_conversion_array_to_dimod():
    pytest.importorskip("dimod")
    arr = np.array([[1.0, 0.5], [0.0, -1.0]])
    ir = QuData.input().from_array(arr)
    bqm = ir.to_dimod_bqm()
    assert bqm is not None
    assert len(bqm.variables) == 2


def test_qudata_input_arithmetic():
    q1 = QuData.input({("q0", "q1"): 1.0})
    q2 = QuData.input({("q0", "q0"): 2.0})
    merged = (q1 + q2).prob
    assert merged[("q0", "q1")] == 1.0
    assert merged[("q0", "q0")] == 2.0


def test_qudata_input_legacy_class_deprecated():
    from qudas.qudata import QuDataInput

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        QuDataInput({("q0", "q0"): 1.0})
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_qudata_output_legacy_class_deprecated():
    from qudas.qudata import QuDataOutput

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        QuDataOutput({"x": 1.0})
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_qudata_output_facade_returns_annealing_output():
    from qudas.annealing import QdAnnealingOutput

    out = QuData.output({"solution": {"q0": 0}, "energy": 0.0})
    assert isinstance(out, QdAnnealingOutput)
    assert "block0" in out.results
