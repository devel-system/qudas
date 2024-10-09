from .qudata_input import QuDataInput
from .qudata_output import QuDataOutput
from typing import Dict, Any

class QuData:
    def __init__(self, input_data: dict = None, output_data: Dict[str, Any] = None, result_type: str = None):
        """
        QuDataInput と QuDataOutput の両方を統合するクラス。
        """
        self.input = QuDataInput(input_data)
        self.output = QuDataOutput(output_data, result_type)