from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, Optional

from qudas.core.base import QdExecutorBase

from .input import QdAnnealingInput
from .output import QdAnnealingOutput


class QdAnnealingExecutor(QdExecutorBase):
    """複数デバイスへの並列実行をサポートするアニーリング用 Executor。"""

    # --------------------------------------------------------------
    # コンストラクタ / パラメータ
    # --------------------------------------------------------------
    def __init__(
        self,
        provider: str = "default",
        provider_config: Optional[Dict[str, Any]] = None,
        provider_map: Optional[Dict[str, str]] = None,
        provider_config_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Parameters
        ----------
        provider : str, optional
            The provider to use for the executor. (e.g. "amplify", "dimod")
        provider_config : dict[str, Any], optional
            The configuration for the provider.
        provider_map : dict[str, str], optional
            The mapping of block labels to providers. (e.g. {"block0": "amplify", "block1": "dimod"})
        provider_config_map : dict[str, dict[str, Any]], optional
            The mapping of block labels to provider configurations. (e.g. {"block0": {"backend": "amplify"}, "block1": {"backend": "dimod"}})
        """
        super().__init__(provider, provider_config, provider_map, provider_config_map)

    # --------------------------------------------------------------
    # パブリック API
    # --------------------------------------------------------------
    def run(
        self, input_data: QdAnnealingInput
    ) -> QdAnnealingOutput:  # noqa: D401 – simple method name
        """単一の :class:`QdAnnealingInput` を実行し、 ``QdAnnealingOutput`` を返却。

        Parameters
        ----------
        input_data : QdAnnealingInput
            実行対象の量子アニーリングブロックを含む入力。

        Returns
        -------
        QdAnnealingOutput
            ブロック名をキー、各 backend の実行結果を値とする辞書を ``results`` として保持します。
        """
        block = input_data.block
        provider = self.resolve_provider(block.label)
        config = self.resolve_provider_config(block.label)
        _, result = self._run_single_block(block, provider, config)
        return QdAnnealingOutput({block.label: result})

    def run_split(
        self, input_data: QdAnnealingInput
    ) -> QdAnnealingOutput:  # noqa: D401 – simple method name
        """与えられた複数ブロックを並列実行し、結果を ``QdAnnealingOutput`` で返却。

        Parameters
        ----------
        input_data : QdAnnealingInput
            実行対象の量子アニーリングブロックを含む入力。

        Returns
        -------
        QdAnnealingOutput
            ブロック名をキー、各 backend の実行結果を値とする辞書を ``results`` として保持します。
        """

        if not hasattr(input_data, "blocks"):
            raise AttributeError("input_data は 'blocks' 属性を持つ必要があります。")

        results: Dict[str, Dict[str, Any]] = {}

        # 並列実行 (CPU バウンドではないため ThreadPoolExecutor で十分)
        with ThreadPoolExecutor() as pool:
            future_map = {
                pool.submit(
                    self._run_single_block,
                    block,
                    self.resolve_provider(block.label),
                    self.resolve_provider_config(block.label),
                ): block.label
                for block in input_data.blocks
            }

            for future in as_completed(future_map):
                label, result = future.result()
                results[label] = result

        return QdAnnealingOutput(results)

    # --------------------------------------------------------------
    # 内部ユーティリティ
    # --------------------------------------------------------------
    def _run_single_block(self, block, provider: str, kwargs: Dict[str, Any]):
        """1 ブロック分の QUBO を指定バックエンドで解く。"""

        if provider == "amplify":
            result = self._run_amplify(block.qubo, **kwargs)
        elif provider == "dimod" or provider == "default":
            result = self._run_dimod(block.qubo, **kwargs)
        else:
            raise NotImplementedError(f"Provider '{provider}' は未サポートです。")

        return block.label, result

    # ------------------------------------------------------------------
    # backend 実装
    # ------------------------------------------------------------------
    @staticmethod
    def _run_amplify(qubo: Dict[Tuple[str, str], float], **kwargs):
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
            return QdAnnealingExecutor._run_naive(qubo, device="amplify(fallback)", **kwargs)  # type: ignore

    @staticmethod
    def _run_dimod(qubo: Dict[Tuple[str, str], float], **kwargs):
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
            return {
                "solution": dict(best.sample),
                "energy": best.energy,
                "device": "dimod",
            }

        except Exception:  # noqa: BLE001 – ImportError or others
            return QdAnnealingExecutor._run_naive(qubo, device="dimod(fallback)", **kwargs)  # type: ignore

    # ------------------------------------------------------------------
    # フォールバック: 単純評価 (すべて 0 に固定)
    # ------------------------------------------------------------------
    @staticmethod
    def _run_naive(qubo: Dict[Tuple[str, str], float], device="naive", **kwargs):
        vars_set = set()
        for key in qubo.keys():
            vars_set.update(key)
        solution = {v: 0 for v in vars_set}
        energy = 0.0
        return {"solution": solution, "energy": energy, "device": device}


# エイリアス
QdAnnExec = QdAnnealingExecutor
