from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from .input_base import QdInputBase
from .output_base import QdOutputBase


class QdExecutorBase(ABC):
    """ゲート／アニーリング方式を問わない Executor の共通インターフェース。"""

    def __init__(
        self,
        provider: str,
        provider_config: Optional[Dict[str, Any]] = None,
        provider_map: Optional[Dict[str, str]] = None,
        provider_config_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Parameters
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
        if provider is None and provider_map is None:
            raise ValueError("provider is required")

        self.provider = provider
        self.provider_config = provider_config or {}
        self.provider_map = provider_map or {}
        self.provider_config_map = provider_config_map or {}

    def resolve_provider(self, label: str) -> str:
        """Resolve the provider for a given block label."""
        return self.provider_map.get(label, self.provider)

    def resolve_provider_config(self, label: str) -> Dict[str, Any]:
        """Resolve the provider configuration for a given block label."""
        return self.provider_config_map.get(label, self.provider_config)

    @abstractmethod
    def run(self, input_data: QdInputBase) -> QdOutputBase:
        """単一の入力を実行し、結果を返します。"""
        ...

    # オプショナル: 分割／並列実行 ----------------------------------------
    def run_split(self, input_data: QdInputBase) -> QdOutputBase:  # noqa: D401
        """大規模入力の分割実行や並列実行を行うオプショナルメソッド。

        デフォルト実装は :class:`NotImplementedError` を送出します。
        必要な場合にサブクラスでオーバーライドしてください。
        """
        raise NotImplementedError(
            "run_split() は必要に応じてサブクラスで実装してください。"
        )


# 下位互換性維持のためのエイリアス
QdExecBase = QdExecutorBase

__all__ = ["QdExecutorBase", "QdExecBase"]
