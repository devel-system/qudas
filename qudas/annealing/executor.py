from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, Optional

from qudas.core.base import QdExecBase

from .input import QdAnnIn
from .output import QdAnnOut


class QuAnnealingExecutor(QdExecBase):
    """複数デバイスへの並列実行をサポートするアニーリング用 Executor。"""

    # --------------------------------------------------------------
    # コンストラクタ / パラメータ
    # --------------------------------------------------------------
    def __init__(self, backend_map: Optional[Dict[str, str]] = None):
        """Parameters
        ----------
        backend_map : dict[str, str], optional
            ブロックラベルをキーに使用するバックエンド名を指定する辞書。
            省略時はすべて ``default`` (= dimod) を用いる。
        """

        self.backend_map = backend_map or {}

    # --------------------------------------------------------------
    # パブリック API
    # --------------------------------------------------------------
    def run(self, input_data: QdAnnIn) -> QdAnnOut:  # noqa: D401 – simple method name
        """与えられた複数ブロックを並列実行し、結果を ``QdAnnOut`` で返却。"""

        if not hasattr(input_data, "blocks"):
            raise AttributeError("input_data は 'blocks' 属性を持つ必要があります。")

        results: Dict[str, Dict[str, Any]] = {}

        # 並列実行 (CPU バウンドではないため ThreadPoolExecutor で十分)
        with ThreadPoolExecutor() as pool:
            future_map = {
                pool.submit(self._run_single_block, block.qubo, self.backend_map.get(block.label, "default"), block.label):
                block.label
                for block in input_data.blocks
            }

            for future in as_completed(future_map):
                block_label, res_dict = future.result()
                results[block_label] = res_dict

        return QdAnnOut(results)

    # --------------------------------------------------------------
    # 内部ユーティリティ
    # --------------------------------------------------------------
    def _run_single_block(self, qubo: Dict[Tuple[str, str], float], backend: str, label: str):
        """1 ブロック分の QUBO を指定バックエンドで解く。"""

        if backend == "amplify":
            result = self._run_amplify(qubo)
        elif backend == "dimod" or backend == "default":
            result = self._run_dimod(qubo)
        else:
            raise NotImplementedError(f"Backend '{backend}' は未サポートです。")

        return label, result

    # ------------------------------------------------------------------
    # backend 実装
    # ------------------------------------------------------------------
    @staticmethod
    def _run_amplify(qubo):
        """Fixstars Amplify を用いて QUBO を解く。(トークン未設定時はフォールバック)"""

        try:
            from amplify import VariableGenerator, Model, FixstarsClient, solve  # type: ignore

            # QUBO dict -> Amplify Poly へ変換
            gen = VariableGenerator()
            variables = {}
            for key in qubo.keys():
                for var_name in key:
                    if var_name not in variables:
                        variables[var_name] = gen.scalar("Binary", name=str(var_name))

            poly = 0
            for key, coeff in qubo.items():
                term = 1
                for var_name in key:
                    term *= variables[var_name]
                poly += coeff * term

            model = Model(poly)

            client = FixstarsClient()
            token = os.getenv("AMPLIFY_TOKEN")
            if token:
                client.token = token
            # timeout 等はデフォルト

            result = solve(model, client)
            solution = {str(k): v for k, v in result.best.values.items()}
            energy = result.best.objective
            return {"solution": solution, "energy": energy, "device": "amplify"}

        except Exception:  # noqa: BLE001 – Any failure → フォールバック
            # Amplify が使えない場合は naive 解法にフォールバック
            return QuAnnealingExecutor._run_naive(qubo, device="amplify(fallback)")

    @staticmethod
    def _run_dimod(qubo):
        """Dimod の ExactSolver で QUBO を解く。dimod が無い場合はフォールバック。"""

        try:
            import dimod  # type: ignore

            linear = {}
            quadratic = {}
            for (i, j), coeff in qubo.items():
                if i == j:
                    linear[i] = coeff
                else:
                    quadratic[(i, j)] = coeff

            bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype="BINARY")
            sampler = dimod.ExactSolver()
            sampleset = sampler.sample(bqm)
            best = sampleset.first
            return {"solution": dict(best.sample), "energy": best.energy, "device": "dimod"}

        except Exception:  # noqa: BLE001 – ImportError or others
            return QuAnnealingExecutor._run_naive(qubo, device="dimod(fallback)")

    # ------------------------------------------------------------------
    # フォールバック: 単純評価 (すべて 0 に固定)
    # ------------------------------------------------------------------
    @staticmethod
    def _run_naive(qubo, device="naive"):
        vars_set = set()
        for key in qubo.keys():
            vars_set.update(key)
        solution = {v: 0 for v in vars_set}
        energy = 0.0
        return {"solution": solution, "energy": energy, "device": device}


# エイリアス
QdAnnExec = QuAnnealingExecutor