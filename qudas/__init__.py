# __init__.py

# バージョン情報をVERSIONファイルから読み込む
import os

with open(os.path.join(os.path.dirname(__file__), "../VERSION")) as version_file:
    __version__ = version_file.read().strip()

from .qudata import QuData
from .pipeline import Pipeline

__all__ = ['QuData', 'Pipeline']
