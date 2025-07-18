# qudata/__init__.py

# QuData, QuDataInput, QuDataOutput, QuDataBase クラスを外部から直接インポートできるようにする
from .qudata import QuData, QuDataInput as _NewQuDataInput, QuDataOutput as _NewQuDataOutput
from .qudata_base import QuDataBase

# エイリアスを公開名に設定
QuDataInput = _NewQuDataInput
QuDataOutput = _NewQuDataOutput

__all__ = ['QuData', 'QuDataInput', 'QuDataOutput', 'QuDataBase']
