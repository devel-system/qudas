import timeit


def adaptive_timeit(func, min_time=0.1, max_repeat=10):
    """
    実行時間に応じて`number`と`repeat`を自動調整して計測を行う。

    Args:
        func (callable): 計測対象の関数
        min_time (float): 1回の試行での最小計測時間（秒）
        max_repeat (int): 最大試行回数

    Returns:
        str: 実行時間の平均と標準偏差を含む計測結果
    """

    # 初期設定
    number = 1
    while True:
        # 初期試行で時間を計測
        time_taken = timeit.timeit(func, number=number)
        if time_taken >= min_time:  # 設定時間を超えたら適切なループ回数とみなす
            break
        number *= 10  # 時間が短すぎる場合、ループ回数を増やす

    # 試行を繰り返して計測
    repeat = min(
        max_repeat, int(max(1, 1 / time_taken))
    )  # `repeat`を自動調整。必ず10以下に設定。
    times = timeit.repeat(func, number=number, repeat=repeat)

    # 平均と標準偏差を計算
    mean_time = sum(times) / repeat
    std_dev = (max(times) - min(times)) / 2

    # 結果を整形
    result = (
        f"{mean_time*1000:.2f} ms ± {std_dev*1000:.2f} ms "
        f"per {number} loops (mean ± std. dev. of {repeat} runs)"
    )
    print(result)
    return mean_time, std_dev, number, repeat
