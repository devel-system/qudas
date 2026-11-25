"""登録済み Step / Block を識別子で復元するレジストリ。

JSON から任意の Python コードは実行しない。未登録 type_id は明示エラー。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class ComponentRegistry:
    """type_id → ファクトリ（params dict → インスタンス）。"""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._classes: Dict[str, Type] = {}

    def register(
        self,
        type_id: str,
        factory: Optional[Callable[..., Any]] = None,
        cls: Optional[Type] = None,
    ) -> None:
        if not type_id or not isinstance(type_id, str):
            raise ValueError("type_id must be a non-empty str")
        if factory is None and cls is None:
            raise ValueError("factory or cls is required")
        if factory is None:

            def _factory(**params: Any) -> Any:
                return cls(**params)  # type: ignore[misc]

            factory = _factory
        self._factories[type_id] = factory
        if cls is not None:
            self._classes[type_id] = cls

    def unregister(self, type_id: str) -> None:
        self._factories.pop(type_id, None)
        self._classes.pop(type_id, None)

    def create(self, type_id: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if type_id not in self._factories:
            raise KeyError(
                f"Unknown component type_id={type_id!r}. "
                "Register it with ComponentRegistry, or use only serializable built-ins."
            )
        params = params or {}
        if not isinstance(params, dict):
            raise TypeError("params must be a dict of JSON-compatible values")
        return self._factories[type_id](**params)

    def has(self, type_id: str) -> bool:
        return type_id in self._factories

    def list_ids(self) -> list:
        return sorted(self._factories.keys())


# グローバル既定レジストリ（組み込みコンポーネントをここに登録）
default_registry = ComponentRegistry()


def register_builtin(type_id: str):
    """デコレータ: クラスを既定レジストリへ登録。"""

    def deco(cls: Type) -> Type:
        default_registry.register(type_id, cls=cls)
        cls._qudas_type_id = type_id  # type: ignore[attr-defined]
        return cls

    return deco
