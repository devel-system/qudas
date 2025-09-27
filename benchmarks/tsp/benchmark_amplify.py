# 巡回セールスマン問題（amplify）
from amplify import VariableGenerator
import networkx as nx


def create_TSP_QUBO_amplify(cities: int = 10):

    # 完全グラフの作成
    G = nx.complete_graph(cities)

    # QUBOの作成
    gen = VariableGenerator()  # 決定変数のジェネレータを作成
    x_arr = gen.array("Binary", shape=(cities, cities))

    # 目的関数
    H = 0
    for u, v in G.edges():
        for a in range(cities - 1):
            H += x_arr[u][a] * x_arr[v][a + 1]

    # 制約関数
    for u in range(cities):
        H += (1 - sum(x_arr[u])) ** 2

        sub_H = 0
        for a in range(cities):
            sub_H += x_arr[a][u]
        H += (1 - sub_H) ** 2

    return H


if __name__ == '__main__':
    from qudas import QuData
    from utils import adaptive_timeit
    import csv

    ##################################
    # ベースのQUBO作成
    ##################################

    # csvに代入するデータ
    csv_data = []

    # 時間計測
    TSP_SIZE = [5, 10, 50, 100]
    for size in TSP_SIZE:
        print(f"TSP size: {size}")
        mean_time, std_dev, number, repeat = adaptive_timeit(
            lambda: create_TSP_QUBO_amplify(size)
        )
        csv_data.append(["-", "-", "-", size, mean_time, std_dev, number, repeat])

    ##################################
    # QuDataから他のデータ形式への変換
    ##################################

    for size in TSP_SIZE:
        # QuData形式に変換
        QUBO = create_TSP_QUBO_amplify(size)
        mean_time, std_dev, number, repeat = adaptive_timeit(
            lambda: QuData.input().from_amplify(QUBO)
        )
        csv_data.append(
            ["amplify", "-", "qudata", size, mean_time, std_dev, number, repeat]
        )
        qudata = QuData.input().from_amplify(QUBO)

        # 他の形式に変換  amplify -> pyqubo, networkx, dimod
        mean_time, std_dev, number, repeat = adaptive_timeit(
            lambda: qudata.to_amplify()
        )
        csv_data.append(
            ["amplify", "qudata", "amplify", size, mean_time, std_dev, number, repeat]
        )
        mean_time, std_dev, number, repeat = adaptive_timeit(lambda: qudata.to_pyqubo())
        csv_data.append(
            ["amplify", "qudata", "pyqubo", size, mean_time, std_dev, number, repeat]
        )
        mean_time, std_dev, number, repeat = adaptive_timeit(
            lambda: qudata.to_networkx()
        )
        csv_data.append(
            ["amplify", "qudata", "networkx", size, mean_time, std_dev, number, repeat]
        )
        mean_time, std_dev, number, repeat = adaptive_timeit(
            lambda: qudata.to_dimod_bqm()
        )
        csv_data.append(
            ["amplify", "qudata", "dimod_bqm", size, mean_time, std_dev, number, repeat]
        )

    # csvに出力
    with open('TSP_result_amplify.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "from",
                "middle",
                "to",
                "size",
                "mean_time (ms)",
                "std_dev (ms)",
                "number",
                "repeat",
            ]
        )
        writer.writerows(csv_data)
