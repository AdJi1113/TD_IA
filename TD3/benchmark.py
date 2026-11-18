"""Compare the search algorithms and generate performance graphs."""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import alphabeta, mcts, minimax
from game import create_board


ALGORITHMS = {
    "minimax": minimax,
    "alphabeta": alphabeta,
    "mcts": mcts,
}


def benchmark(board, algorithm, **kwargs):
    """Run an algorithm and return its main performance measurements."""
    if isinstance(algorithm, str):
        name = algorithm.lower()
        if name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        function = ALGORITHMS[name]
    else:
        function = algorithm
        name = function.__name__

    start = time.perf_counter()
    result = function(board, **kwargs)
    elapsed = time.perf_counter() - start
    return {
        "algorithm": name,
        "move": result["move"],
        "score": result["score"],
        "nodes_explored": result["nodes_explored"],
        "time": elapsed,
    }


def print_table(results):
    headers = ("Algorithm", "Parameter", "Move", "Score", "Nodes", "Time (s)")
    print(f"{headers[0]:<12} {headers[1]:<14} {headers[2]:>6} {headers[3]:>10} {headers[4]:>10} {headers[5]:>10}")
    print("-" * 68)
    for result in results:
        print(
            f"{result['algorithm']:<12} {result['parameter']:<14} "
            f"{result['move']:>6} {result['score']:>10} "
            f"{result['nodes_explored']:>10} {result['time']:>10.6f}"
        )


def create_graphs(results, output_directory):
    output_directory.mkdir(exist_ok=True)

    minimax_results = [result for result in results if result["algorithm"] == "minimax"]
    alphabeta_results = [result for result in results if result["algorithm"] == "alphabeta"]

    plt.figure(figsize=(8, 5))
    plt.plot(
        [result["depth"] for result in minimax_results],
        [result["nodes_explored"] for result in minimax_results],
        marker="o",
        label="Minimax",
    )
    plt.plot(
        [result["depth"] for result in alphabeta_results],
        [result["nodes_explored"] for result in alphabeta_results],
        marker="o",
        label="Alpha-Beta",
    )
    plt.xlabel("Depth")
    plt.ylabel("Nodes explored")
    plt.title("Nodes explored by search depth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "minimax_alphabeta_nodes.png", dpi=150)
    plt.close()

    mcts_results = [result for result in results if result["algorithm"] == "mcts"]
    plt.figure(figsize=(8, 5))
    plt.plot(
        [result["simulations"] for result in mcts_results],
        [result["time"] for result in mcts_results],
        marker="o",
    )
    plt.xlabel("Number of simulations")
    plt.ylabel("Execution time (seconds)")
    plt.title("MCTS execution time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / "mcts_time.png", dpi=150)
    plt.close()


def main():
    board = create_board()
    results = []

    for depth in range(1, 6):
        for algorithm in (minimax, alphabeta):
            result = benchmark(board, algorithm, depth=depth)
            result["depth"] = depth
            result["parameter"] = f"depth={depth}"
            results.append(result)

    for simulations in (100, 500, 1000, 2000):
        result = benchmark(board, mcts, num_simulations=simulations, seed=42)
        result["simulations"] = simulations
        result["parameter"] = f"simulations={simulations}"
        results.append(result)

    print_table(results)
    create_graphs(results, Path(__file__).parent / "benchmark_results")
    print("\nGraphs saved in benchmark_results/")


if __name__ == "__main__":
    main()
