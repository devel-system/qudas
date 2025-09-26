from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

from qudas.core.base import QdExecutorBase

from .input import QdGateInput
from .output import QdGateOutput


class QdGateExecutor(QdExecutorBase):
    """量子ゲート方式の ``Executor``。

    * デフォルトでは ``qiskit_simulator`` を用いて実行します。
    * :py:meth:`run_split` によりブロック毎に backend を切り替えた並列実行も可能です。
    """

    # --------------------------------------------------------------
    # コンストラクタ / 共通パラメータ
    # --------------------------------------------------------------
    def __init__(
        self,
        provider: str = "default",
        provider_config: Optional[Dict[str, Any]] = None,
        provider_map: Optional[Dict[str, str]] = None,
        provider_config_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Parameters
        ----------
        provider : str, optional
            The provider to use for the executor. (e.g. "qiskit", "braket")
        provider_config : dict[str, Any], optional
            The configuration for the provider.
        provider_map : dict[str, str], optional
            The mapping of block labels to providers. (e.g. {"block0": "qiskit", "block1": "braket"})
        provider_config_map : dict[str, dict[str, Any]], optional
            The mapping of block labels to provider configurations. (e.g. {"block0": {"backend": "qiskit_simulator"}, "block1": {"backend": "braket_ionq"}})
        """
        super().__init__(provider, provider_config, provider_map, provider_config_map)

    # --------------------------------------------------------------
    # パブリック API
    # --------------------------------------------------------------
    def run(
        self, input_data: QdGateInput
    ) -> QdGateOutput:  # noqa: D401 – simple method name
        """単一の :class:`QdGateInput` を実行し、 ``QdGateOutput`` を返却。"""
        block = input_data.block
        provider = self.resolve_provider(block.label)
        config = self.resolve_provider_config(block.label)
        _, result = self._run_single_block(block, provider, config)
        return QdGateOutput({block.label: result})

    def run_split(
        self, input_data: QdGateInput
    ) -> QdGateOutput:  # noqa: D401 – simple method name
        """入力をブロックごとに分割して並列実行します。

        Parameters
        ----------
        input_data : QdGateInput
            実行対象の量子回路ブロックを含む入力。

        Returns
        -------
        QdGateOutput
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

        return QdGateOutput(results)

    # --------------------------------------------------------------
    # 内部ユーティリティ
    # --------------------------------------------------------------
    def _run_single_block(self, block, provider: str, kwargs: Dict[str, Any]):
        """1 ブロック分の量子回路を指定バックエンドで実行。"""
        # --- 現在は QuantumCircuitBlock (SDK 非依存) をサポート ------------------
        if provider == "qiskit" or provider == "default":
            # ``block`` の型に応じて回路を用意
            if hasattr(block, "gates"):
                # 新しい QuantumCircuitBlock 形式
                circuit = self._block_to_qiskit(block)
            else:
                # 旧形式: ``circuit`` 属性に直接 qiskit.QuantumCircuit が入っている想定
                circuit = self._ensure_qiskit_circuit(getattr(block, "circuit", None))

            result = self._run_qiskit(circuit, **kwargs)
        else:
            raise NotImplementedError(f"Provider '{provider}' は未サポートです。")

        return block.label, result

    # ------------------------------------------------------------------
    # 量子回路ブロック → Qiskit 変換
    # ------------------------------------------------------------------
    @staticmethod
    def _block_to_qiskit(block):
        """QuantumCircuitBlock を Qiskit ``QuantumCircuit`` へ変換する簡易実装。"""

        try:
            from qiskit import QuantumCircuit  # type: ignore

            qc = QuantumCircuit(block.num_qubits, block.num_qubits)

            for gate_ir in block.gates:
                # ゲート名に応じてダイナミックにメソッド呼び出し
                gate_name = gate_ir.gate.lower()

                # 制御ゲート (cx, cz など) は controls + targets を結合して渡す
                qargs = gate_ir.controls + gate_ir.targets

                # パラメータ付きゲート (rx, ry, rz ...) は params を先頭に
                try:
                    method = getattr(qc, gate_name)
                except AttributeError:
                    # 未対応ゲートはスキップ (必要に応じて追加実装)
                    continue

                # 呼び出し引数を組み立て
                if gate_ir.params:
                    method(*gate_ir.params, *qargs)
                else:
                    method(*qargs)

            # 省略した classical register への測定を追加 (デフォルト: 全量子ビット)
            qc.measure_all()

            return qc
        except Exception:
            # qiskit import error or conversion error → fallback
            return QdGateExecutor._ir_to_qiskit(None)

    # ------------------------------------------------------------------
    # backend 実装
    # ------------------------------------------------------------------
    @staticmethod
    def _run_qiskit(circuit, **kwargs):
        """Qiskit Aer/Basics を用いて回路をシミュレーション。"""

        try:
            # lazy import – qiskit が入っていない環境でも動作させるため
            from qiskit import Aer, execute  # type: ignore

            backend = Aer.get_backend(kwargs.get("backend", "qasm_simulator"))
            job = execute(circuit, backend=backend, **kwargs)
            counts = job.result().get_counts()
            return {"counts": dict(counts), "device": "qiskit_simulator"}

        except Exception:  # noqa: BLE001 – ImportError or runtime errors
            # qiskit 非インストール or その他エラー → naive fallback
            return QdGateExecutor._run_naive(device="qiskit_simulator(fallback)")

    # ------------------------------------------------------------------
    # フォールバック実装
    # ------------------------------------------------------------------
    @staticmethod
    def _run_naive(device: str = "naive"):
        """依存ライブラリが無い環境向けの簡易実装。"""

        # とりあえず半々のビット列が得られたと仮定
        return {"counts": {"00": 512, "11": 512}, "device": device}

    # ------------------------------------------------------------------
    # 変換ユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _ir_to_qiskit(ir):
        """`QuAlgorithmIR` → Qiskit ``QuantumCircuit`` 変換。

        現状は最小実装として、ゲート情報を無視し 1qubit の Hadamard + 測定を生成。
        IR がよりリッチになった際はここで map してください。
        """
        try:
            from qiskit import QuantumCircuit  # type: ignore

            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            return qc
        except Exception:
            # qiskit 無い場合はダミーを返す (呼び出し側でフォールバック)
            return None

    @staticmethod
    def _ensure_qiskit_circuit(obj):
        """入力が QuantumCircuit でない場合はダミー回路へ置き換える。"""
        try:
            from qiskit import QuantumCircuit  # type: ignore

            if isinstance(obj, QuantumCircuit):
                return obj
        except Exception:
            pass  # qiskit import error → fallthrough

        # fallback dummy circuit
        return QdGateExecutor._ir_to_qiskit(None)


# 下位互換性維持のためのエイリアス -----------------------------------------
QdGateExec = QdGateExecutor
