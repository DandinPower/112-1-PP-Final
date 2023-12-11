def benchmark_example():
    """
    An example of how to benchmark a single test configuration.
    """
    from src.utils import SparseMatrixTestConfiguration
    from src.benchmark import show_benchmark_results, benchmark_by_config_list, generate_benchmark_configurations
    config_list = generate_benchmark_configurations(size_start=5, size_end=500, size_step=50, density_start=1, density_end=10, density_step=4)
    results = benchmark_by_config_list(config_list, num_runs=10)
    show_benchmark_results(results)

def main():
    benchmark_example()

if __name__ == '__main__':
    main()