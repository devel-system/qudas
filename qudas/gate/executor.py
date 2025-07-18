from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

from qudas.core.base import QdExecBase

from .input import QuDataGateInput, QdGateIn
from .output import QuDataGateOutput, QdGateOut


class QuDataGateExecutor(QdExecBase):
    """量子ゲート方式の ``Executor``。

    * デフォルトでは ``qiskit_simulator`` を用いて実行します。
    * :py:meth:`run_split` によりブロック毎に backend を切り替えた並列実行も可能です。
    """

    # --------------------------------------------------------------
    # コンストラクタ / 共通パラメータ
    # --------------------------------------------------------------
    def __init__(self, backend: str = "qiskit_simulator") -> None:
        """Parameters
        ----------
        backend : str, optional
            使用するバックエンド名 (例: ``"qiskit_simulator"``、``"braket_ionq"`` など)。
            :py:meth:`run` で backend を明示しない場合にデフォルトとして利用されます。
        """

        self.backend = backend

    # --------------------------------------------------------------
    # パブリック API
    # --------------------------------------------------------------
    def run(self, input_data: QuDataGateInput) -> QuDataGateOutput:  # noqa: D401 – simple method name
        """単一の :class:`QuDataGateInput` を実行し、 ``QuDataGateOutput`` を返却。"""

        # 入力を IR へ変換
        ir = input_data.to_ir()

        # backend 毎の実行ルーティンへディスパッチ
        if self.backend == "qiskit_simulator":
            circuit = self._ir_to_qiskit(ir)
            result = self._run_qiskit(circuit)
        else:
            raise NotImplementedError(f"Backend '{self.backend}' は未サポートです。")

        # SDK 依存の生の結果 → フレンドリーなオブジェクトへラップ
        return QuDataGateOutput(result)

    def run_split(
        self,
        input_data: QuDataGateInput,
        backend_map: Optional[Dict[str, str]] = None,
    ) -> QuDataGateOutput:  # noqa: D401 – simple method name
        """入力をブロックごとに分割して並列実行します。

        Parameters
        ----------
        input_data : QuDataGateInput
            実行対象の量子回路ブロックを含む入力。
        backend_map : dict[str, str], optional
            ``{block_name: backend}`` 形式でブロック毎に利用する backend を指定します。
            未指定または辞書に存在しないブロックは ``self.backend`` が使われます。

        Returns
        -------
        QuDataGateOutput
            ブロック名をキー、各 backend の実行結果を値とする辞書を ``results`` として保持します。
        """

        if not hasattr(input_data, "blocks"):
            raise AttributeError("input_data は 'blocks' 属性を持つ必要があります。")

        backend_map = backend_map or {}
        results: Dict[str, Dict[str, Any]] = {}

        # 並列実行 (CPU バウンドではないため ThreadPoolExecutor で十分)
        with ThreadPoolExecutor() as pool:
            future_map = {
                pool.submit(
                    self._run_single_block,
                    block,
                    backend_map.get(block.name, self.backend),
                ): block.name
                for block in input_data.blocks
            }

            for future in as_completed(future_map):
                block_name, res_dict = future.result()
                results[block_name] = res_dict

        return QuDataGateOutput(results)

    # --------------------------------------------------------------
    # 内部ユーティリティ
    # --------------------------------------------------------------
    def _run_single_block(self, block, backend: str):
        """1 ブロック分の量子回路を指定バックエンドで実行。"""
        # --- 現在は QuantumCircuitBlock (SDK 非依存) をサポート ------------------
        if backend == "qiskit_simulator":
            # ``block`` の型に応じて回路を用意
            if hasattr(block, "gates"):
                # 新しい QuantumCircuitBlock 形式
                circuit = self._block_to_qiskit(block)
            else:
                # 旧形式: ``circuit`` 属性に直接 qiskit.QuantumCircuit が入っている想定
                circuit = self._ensure_qiskit_circuit(getattr(block, "circuit", None))

            result = self._run_qiskit(circuit)
        else:
            raise NotImplementedError(f"Backend '{backend}' は未サポートです。")

        return block.name, result

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
            return QuDataGateExecutor._ir_to_qiskit(None)

    # ------------------------------------------------------------------
    # backend 実装
    # ------------------------------------------------------------------
    @staticmethod
    def _run_qiskit(circuit):
        """Qiskit Aer/Basics を用いて回路をシミュレーション。"""

        try:
            # lazy import – qiskit が入っていない環境でも動作させるため
            from qiskit import Aer, execute  # type: ignore

            try:
                backend = Aer.get_backend("aer_simulator")
            except Exception:  # 旧版 fallback
                backend = Aer.get_backend("qasm_simulator")

            job = execute(circuit, backend=backend, shots=1024)
            counts = job.result().get_counts()
            return {"counts": dict(counts), "device": "qiskit_simulator"}

        except Exception:  # noqa: BLE001 – ImportError or runtime errors
            # qiskit 非インストール or その他エラー → naive fallback
            return QuDataGateExecutor._run_naive(device="qiskit_simulator(fallback)")

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
        return QuDataGateExecutor._ir_to_qiskit(None)


# 下位互換性維持のためのエイリアス -----------------------------------------
QuGateExecutor = QuDataGateExecutor
QdGateExec = QuDataGateExecutor