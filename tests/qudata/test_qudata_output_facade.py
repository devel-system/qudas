"""WP-1.4: QuData.output() / QdAnnealingOutput 現行 API の facade テスト。

レガシー QuDataOutput の ``.result`` / ``.qubo`` は期待しない。
ブロック形式 ``results[block]['solution'|'energy']`` を正とする。
"""

from __future__ import annotations

import numpy as np
import pytest

from qudas import QuData
from qudas.annealing import QdAnnealingOutput


def test_output_from_pulp_uses_block_contract():
    pulp = pytest.importorskip("pulp")
    from pulp import LpMinimize, LpProblem, LpVariable

    prob = LpProblem("Test Problem", LpMinimize)
    x = LpVariable("x", lowBound=0, upBound=1, cat="Binary")
    y = LpVariable("y", lowBound=0, upBound=1, cat="Binary")
    prob += 2 * x - y
    prob.solve()

    out = QuData.output().from_pulp(prob)
    assert isinstance(out, QdAnnealingOutput)
    block = out.results["block0"]
    assert block["solution"] == {"x": 0.0, "y": 1.0}
    assert block["energy"] == -1.0
    assert block.get("device") == "pulp"


def test_output_from_dimod_uses_block_contract():
    dimod = pytest.importorskip("dimod")
    qubo = {("q0", "q0"): 2, ("q1", "q1"): -1, ("q2", "q2"): -1}
    sampleset = dimod.ExactSolver().sample_qubo(qubo)

    out = QuData.output().from_dimod(sampleset)
    block = out.results["block0"]
    assert block["solution"] == {"q0": 0.0, "q1": 1.0, "q2": 1.0}
    assert block["energy"] == -2.0
    assert block.get("device") == "dimod"
    assert "statistics" in block


def test_output_to_dimod_from_facade_dict():
    dimod = pytest.importorskip("dimod")
    out = QuData.output(
        {
            "solution": {"q0": 0.0, "q1": 1.0, "q2": 1.0},
            "energy": -2,
        }
    )
    sampleset = out.to_dimod()
    assert isinstance(sampleset, dimod.SampleSet)
    assert dict(sampleset.first.sample) == {"q0": 0, "q1": 1, "q2": 1}
    assert sampleset.first.energy == -2


def test_output_from_scipy_uses_block_contract():
    pytest.importorskip("scipy")
    pytest.importorskip("sympy")
    from scipy.optimize import Bounds, minimize
    from sympy import lambdify, symbols

    q0, q1, q2 = symbols("q0 q1 q2")
    f = lambdify([q0, q1, q2], 2 * q0 - q1 - q2, "numpy")
    res = minimize(
        lambda qv: f(qv[0], qv[1], qv[2]),
        [0.5, 0.5, 0.5],
        method="SLSQP",
        bounds=Bounds([0, 0, 0], [1, 1, 1]),
    )

    out = QuData.output().from_scipy(res)
    block = out.results["block0"]
    # 現行 API は変数名を q0,q1,... とする（レガシー result キーとは別）
    np.testing.assert_allclose(list(block["solution"].values()), res.x, atol=1e-6)
    assert block["energy"] == pytest.approx(float(res.fun))
    assert block.get("device") == "scipy"


def test_output_has_no_legacy_qubo_attr():
    out = QuData.output({"solution": {"q0": 0}, "energy": 0.0})
    assert not hasattr(out, "qubo")
    assert "block0" in out.results
