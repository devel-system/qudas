# qudas/pipeline/base.py
from __future__ import annotations


import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict


class QdBaseEstimator:
    """
    sklearn 互換のパラメータ操作と、実行時コンテキスト (context) を提供する基底クラス。

    - get_params(deep=True) / set_params(**params)
    - set_context(context: dict) / get_context()
    - 旧 API 互換: set_global_params()/get_global_params() は context["params"] と連動
    """

    def __init__(self, **kwargs):
        # __init__ 引数をそのまま属性に反映（sklearn 互換）
        for k, v in kwargs.items():
            setattr(self, k, v)
        # 実行時の共有情報（乱数種・実行環境・logger・global params 等）
        self._context: Dict[str, Any] = {"params": {}}

    # ---- sklearn 互換 API -------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        自身の __init__ シグネチャに基づき、現在のパラメータ辞書を返す。
        """
        try:
            sig = inspect.signature(self.__init__)
        except (TypeError, ValueError):
            # __init__ が introspect 不能な場合のフォールバック
            return {}

        out: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if hasattr(self, name):
                out[name] = getattr(self, name)
            else:
                # デフォルト引数をそのまま返す（属性未設定の場合）
                if param.default is not inspect._empty:
                    out[name] = param.default
        return out

    def set_params(self, **params) -> "QdBaseEstimator":
        """
        self の属性としてパラメータをセットして返す（メソッドチェーン可能）。
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- 実行時コンテキスト ----------------------------------------------
    def set_context(self, context: Dict[str, Any]) -> None:
        """
        パイプラインから注入される実行時コンテキストを保存する。
        例: {"params": {...}, "rng": np.random.Generator(...), "logger": ...}
        """
        self._context = context

    def get_context(self) -> Dict[str, Any]:
        """現在の実行コンテキストを返す。"""
        return self._context

    # ---- 旧 API 互換 (global_params) -------------------------------------
    def set_global_params(self, params: Dict[str, Any]) -> None:
        """
        旧 API 互換。context["params"] に書き込むシンタックスシュガー。
        """
        if not isinstance(self._context, dict):
            self._context = {}
        self._context["params"] = params

    def get_global_params(self) -> Dict[str, Any]:
        """
        旧 API 互換。context["params"] を返す。
        """
        if not isinstance(self._context, dict):
            return {}
        return self._context.get("params", {})  # type: ignore[return-value]


class QdBaseStep(QdBaseEstimator, ABC):
    """
    すべてのパイプライン Step の基底クラス。
    必須メソッドはなく、必要に応じて以下を実装する：

      - fit(self, X, y=None) -> self
      - transform(self, X) -> Any
      - predict(self, X) -> Any
      - optimize(self, X=None, y=None) -> Any
    """

    # デフォルト実装（何もしない / 未実装は AttributeError）
    def fit(self, X: Any, y: Any = None) -> "QdBaseStep":  # sklearn 互換で self を返す
        return self

    def transform(self, X: Any) -> Any:
        return X

    def predict(self, X: Any) -> Any:
        raise AttributeError(f"{self.__class__.__name__} has no predict().")

    def optimize(self, X: Any = None, y: Any = None) -> Any:
        raise AttributeError(f"{self.__class__.__name__} has no optimize().")

class OptimizerStep(QdBaseStep, ABC):
    """
    最適化を担う Step の抽象クラス。
    ユーザーは本クラスを継承して FMQA/VQE などのアルゴリズムを実装する。
    """

    @abstractmethod
    def optimize(self, X: Any = None, y: Any = None) -> Any:
        """
        任意の入力 X（前段の transform 出力）と y を受け、最適化結果を返す。
        """
        ...


# ---- 旧コード互換のためのエイリアス --------------------------------------
# 既存コードが `from .base import BaseStep` を前提としていても動作するように。
BaseStep = QdBaseStep

__all__ = [
    "QdBaseEstimator",
    "QdBaseStep",
    "OptimizerStep",
    "BaseStep",
]