"""QdAnnealingOutput.from_amplify の Amplify SDK バージョン互換テスト。"""

from __future__ import annotations

import unittest

from qudas.annealing.output import QdAnnealingOutput


class _Solution:
    def __init__(self, values, *, objective=None, energy=None):
        self.values = values
        if objective is not None:
            self.objective = objective
        if energy is not None:
            self.energy = energy


class _Result:
    """Amplify Result の最小モック。"""

    def __init__(self, solutions, *, energies=None, with_best=True):
        self.solutions = solutions
        if with_best and solutions:
            self.best = solutions[0]
        if energies is not None:
            self.energies = energies

    def __len__(self):
        return len(self.solutions)

    def __iter__(self):
        return iter(self.solutions)

    def __getitem__(self, key):
        return self.solutions[key]


class TestFromAmplifyCompat(unittest.TestCase):
    def test_v1_style_objective_and_solutions(self):
        """Amplify v1: best.objective + solutions（energies 属性なし）。"""
        result = _Result(
            [
                _Solution({"q_0": 0, "q_1": 1}, objective=-1.0),
                _Solution({"q_0": 1, "q_1": 1}, objective=0.0),
            ]
        )
        out = QdAnnealingOutput.from_amplify(result)
        block = out.results["block0"]
        self.assertEqual(block["solution"], {"q_0": 0, "q_1": 1})
        self.assertEqual(block["energy"], -1.0)
        self.assertEqual(block["energies"], [-1.0, 0.0])
        self.assertEqual(block["statistics"]["bitstring"]["unique"], 2)

    def test_legacy_energies_attribute(self):
        """旧 API / ラッパー: result.energies がある場合はそれを優先。"""
        best = _Solution({"x": 1}, objective=-2.0)
        result = _Result([best], energies=[-2.0, -1.5, 0.0])
        out = QdAnnealingOutput.from_amplify(result)
        block = out.results["block0"]
        self.assertEqual(block["energies"], [-2.0, -1.5, 0.0])
        self.assertEqual(block["energy"], -2.0)

    def test_v0_style_energy_attribute(self):
        """Amplify v0 系: solution.energy（objective なし）。"""
        result = _Result(
            [
                _Solution({"a": 0, "b": 1}, energy=-3.0),
                _Solution({"a": 1, "b": 0}, energy=-1.0),
            ]
        )
        out = QdAnnealingOutput.from_amplify(result)
        block = out.results["block0"]
        self.assertEqual(block["energy"], -3.0)
        self.assertEqual(block["energies"], [-3.0, -1.0])
        self.assertEqual(block["solution"], {"a": 0, "b": 1})

    def test_solutions_only_without_best(self):
        """best が無く solutions だけの結果。"""
        result = _Result(
            [_Solution({"q0": 1}, objective=1.5)],
            with_best=False,
        )
        self.assertFalse(hasattr(result, "best"))
        out = QdAnnealingOutput.from_amplify(result)
        self.assertEqual(out.results["block0"]["energy"], 1.5)

    def test_iterable_result_without_solutions_attr(self):
        """solutions 属性が無く、Result 自体がイテレート可能な場合。"""

        class _IterableResult:
            def __init__(self, solutions):
                self.best = solutions[0]
                self._solutions = solutions

            def __iter__(self):
                return iter(self._solutions)

            def __len__(self):
                return len(self._solutions)

        result = _IterableResult(
            [
                _Solution({"q0": 0}, objective=2.0),
                _Solution({"q0": 1}, objective=3.0),
            ]
        )
        out = QdAnnealingOutput.from_amplify(result)
        self.assertEqual(out.results["block0"]["energies"], [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
