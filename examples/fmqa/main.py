# qudas & pipline
from qudas.pipeline import Pipeline
from torch_fmqa import TorchFMQA
from anneal_fmqa import AnnealFMQA
from pipe_iteration import PipeIteration
from utils import make_blackbox_func, init_training_data

# module
import torch

# ENV
from dotenv import load_dotenv
import os

# .envファイルの内容をロード
load_dotenv()

# envファイルからtokenを取得
token = os.getenv("AMPLIFY_TOKEN")

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
global_parameters = {'v': v, 'w': w, 'w0': w0, 'blackbox': blackbox, 'd': d}

# pipeline
steps = [('TorchFMQA', TorchFMQA()), ('AnnealFMQA', AnnealFMQA(token))]

pipe = Pipeline(steps, iterator=PipeIteration(loop_num=N))
pipe.set_global_params(global_parameters)

# 最適化
result = pipe.optimize(x, y)
print(f"{result=}")
# result={'TorchFMQA': None, 'AnnealFMQA': array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
#        1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
#        0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
#        1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
#        1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0])}
