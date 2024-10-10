# qudas & pipline
from qudas.pipeline import Pipeline
from torch_fmqa import TorchFMQA
from anneal_fmqa import AnnealFMQA
from pipe_iteration import PipeIteration
from utils import make_blackbox_func, init_training_data

# module
import torch

# 適当な関数を作成 (d次元, y = xQx)
d = 100
blackbox = make_blackbox_func(d)

# 初期教師データの数
N0 = 60
x, y = init_training_data(d, N0, blackbox)
print(f"{x.shape=}, {y.shape=}")

# FMQA サイクルの実行回数
N = 10

# 初期パラメータ
k = 10
v = torch.randn((d, k), requires_grad=True)
w = torch.randn((d,), requires_grad=True)
w0 = torch.randn((), requires_grad=True)
parameters = {'v': v, 'w': w, 'w0': w0}

# pipeline
steps = [
    ('TorchFMQA', TorchFMQA()),
    ('AnnealFMQA', AnnealFMQA(blackbox, d, token="AE/HaqGh1iuFMEennXk10xS1LCgld8D18oC")),
    ('pipeIteration', PipeIteration(blackbox, d, loop_num=N))
]

pipe = Pipeline(steps)
pipe.set_global_params(parameters)

# 最適化
result = pipe.optimize(x, y)
print(f"{result=}")
# print(f"{pipe.get_grobal_params()=}")