"""テスト共有ユーティリティ（pytest 対象外）。"""

from __future__ import annotations


def qubo_dicts_equal(a: dict, b: dict) -> bool:
    """QUBO dict をキー集合無視で比較する。"""
    if len(a) != len(b):
        return False
    for k1, v1 in a.items():
        if not any(set(k1) == set(k2) and v1 == v2 for k2, v2 in b.items()):
            return False
    return True
