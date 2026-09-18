"""WP-1: QuData facade / QdAnnealingIR 変換契約の正のテスト。

Executor は呼ばない。レガシー QuDataInput 直叩きは tests/qudata/test_input.py に残す。
"""

from __future__ import annotations

import warnings
from pathlib import Path

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

_DATA = Path(__file__).resolve().parent.parent / "data"


from tests.helpers import qubo_dicts_equal as _dicts_equal


def test_package_version_is_not_placeholder():
    import qudas

    assert qudas.__version__ == "0.2.6"


def test_core_new_names_are_exported():
    assert QdInputBase is QdInBase
    assert QdOutputBase is QdOutBase
    assert QdExecutorBase is QdExecBase


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


def test_qudata_input_from_array_detailed():
    arr = np.array(
        [
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, -1],
        ]
    )
    ir = QuData.input().from_array(arr)
    expected = {
        ("q_0", "q_0"): 1,
        ("q_0", "q_1"): 1,
        ("q_1", "q_1"): 2,
        ("q_2", "q_2"): -1,
    }
    assert _dicts_equal(ir.prob, expected)
    np.testing.assert_array_equal(ir.to_array(), arr)


def test_qudata_input_from_pulp_and_to_pulp():
    pulp = pytest.importorskip("pulp")
    q0 = pulp.LpVariable("q0", lowBound=0, upBound=1, cat="Binary")
    q1 = pulp.LpVariable("q1", lowBound=0, upBound=1, cat="Binary")
    problem = pulp.LpProblem("QUBO", pulp.LpMinimize)
    problem += 2 * q0 - q1

    ir = QuData.input().from_pulp(problem)
    assert _dicts_equal(ir.prob, {("q0", "q0"): 2, ("q1", "q1"): -1})

    roundtrip = ir.to_pulp()
    assert isinstance(roundtrip, pulp.LpProblem)
    assert str(roundtrip.objective) == str(problem.objective)


def test_qudata_input_from_dimod_bqm_roundtrip():
    dimod = pytest.importorskip("dimod")
    bqm = dimod.BinaryQuadraticModel({"q2": -1}, {("q0", "q1"): 1}, vartype="BINARY")
    ir = QuData.input().from_dimod_bqm(bqm)
    assert _dicts_equal(ir.prob, {("q0", "q1"): 1, ("q2", "q2"): -1})
    assert ir.to_dimod_bqm() == bqm


def test_qudata_input_from_pandas():
    pd = pytest.importorskip("pandas")
    array = np.array(
        [
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, -1],
        ]
    )
    df = pd.DataFrame(array, columns=["q0", "q1", "q2"], index=["q0", "q1", "q2"])
    ir = QuData.input().from_pandas(df)
    expected = {("q0", "q0"): 1, ("q0", "q1"): 1, ("q1", "q1"): 2, ("q2", "q2"): -1}
    assert _dicts_equal(ir.prob, expected)


def test_qudata_input_from_networkx():
    nx = pytest.importorskip("networkx")
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    ir = QuData.input().from_networkx(g)
    expected = {("q_0", "q_1"): 1, ("q_1", "q_2"): 1, ("q_0", "q_2"): 1}
    assert _dicts_equal(ir.prob, expected)
    out_g = ir.to_networkx()
    assert isinstance(out_g, nx.Graph)
    # to_networkx は変数名文字列をノードに使う（入力の整数ノードとは別契約）
    assert set(out_g.nodes()) == {"q_0", "q_1", "q_2"}
    assert {frozenset(e) for e in out_g.edges()} == {
        frozenset({"q_0", "q_1"}),
        frozenset({"q_1", "q_2"}),
        frozenset({"q_0", "q_2"}),
    }


def test_qudata_input_from_sympy_and_to_sympy():
    sympy = pytest.importorskip("sympy")
    q0, q1, q2 = sympy.symbols("q0 q1 q2")
    expr = q0 * q1 - q2**2
    ir = QuData.input().from_sympy(expr)
    assert _dicts_equal(ir.prob, {("q0", "q1"): 1, ("q2", "q2"): -1})
    # to_sympy は対角項を q**2 ではなく q として戻す（現行契約）
    assert ir.to_sympy() == q0 * q1 - q2


def test_qudata_input_from_json():
    ir = QuData.input().from_json(str(_DATA / "qudata.json"))
    expected = {
        ("q0", "q0"): 1.0,
        ("q0", "q1"): 1.0,
        ("q1", "q1"): -1.0,
        ("q2", "q2"): 2.0,
    }
    assert _dicts_equal(ir.prob, expected)


def test_qudata_input_arithmetic():
    q1 = QuData.input({("q0", "q1"): 1.0})
    q2 = QuData.input({("q0", "q0"): 2.0})
    merged = (q1 + q2).prob
    assert merged[("q0", "q1")] == 1.0
    assert merged[("q0", "q0")] == 2.0


def test_qudata_input_pow():
    ir = QuData.input({("q0", "q1"): 1.0, ("q2", "q2"): -1.0})
    result = ir**2
    expected = {("q0", "q1"): 1.0, ("q0", "q2", "q1"): -2.0, ("q2", "q2"): 1.0}
    assert _dicts_equal(result.prob, expected)


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
