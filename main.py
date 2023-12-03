from src.benchmark import benchmark_functions_single_test, benchmark_functions_multiple_test, show_benchmark_results
from src.utils import SparseMatrixTestConfiguration

def main():
    results = []
    for size in range(5, 21, 5):
        for density in [i/10 for i in range(2, 10, 7)]:
            configurations = []
            for density_2 in [i/10 for i in range(2, 10, 7)]:
                configurations.append(SparseMatrixTestConfiguration(size, size, density, size, size, density_2))
            benchmark_functions_multiple_test(configurations, results)
    show_benchmark_results(results)

if __name__ == '__main__':
    main()