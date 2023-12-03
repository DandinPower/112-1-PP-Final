def single_benchmark_example():
    """
    An example of how to benchmark a single test configuration.
    """
    from src.benchmark import benchmark_functions_single_test, SparseMatrixTestConfiguration
    benchmark_functions_single_test(SparseMatrixTestConfiguration(5, 5, 0.1, 5, 5, 0.1), 1)

def multiple_benchmark_example():
    """
    An example of how to benchmark multiple test configurations.
    """
    from src.benchmark import generate_and_benchmark_configurations, show_benchmark_results
    results = generate_and_benchmark_configurations(size_start=5, size_end=500, size_step=50, density_start=1, density_end=10, density_step=4)
    show_benchmark_results(results)

def main():
    # single_benchmark_example()
    multiple_benchmark_example()

if __name__ == '__main__':
    main()