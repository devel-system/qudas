# pipeline/__init__.py

from .pipeline import QdPipeline

# 旧コード・テスト互換
Pipeline = QdPipeline

__all__ = ["QdPipeline", "Pipeline"]
