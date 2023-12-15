def simple_test():
    from src.utils     import generate_sparse_matrix
    from src.extension import ExtensionHandler
    from src.benchmark import SparseMatrixTestConfiguration
    config = SparseMatrixTestConfiguration(10, 10, 0.3, 10, 10, 0.3)
    sparse_matrix_0 = generate_sparse_matrix(config.A_row, config.A_col, density=config.A_density)
    sparse_matrix_1 = generate_sparse_matrix(config.B_row, config.B_col, density=config.B_density)
    print("=" * 50 + "[sparse_mm]" + "=" * 50)
    ExtensionHandler.sparse_mm(sparse_matrix_0, sparse_matrix_1)
    print("=" * 50 + "[openmp_sparse_mm]" + "=" * 50)
    ExtensionHandler.openmp_sparse_mm(sparse_matrix_0, sparse_matrix_1)

def single_benchmark_example():
    """
    An example of how to benchmark a single test configuration.
    """
    from src.benchmark import benchmark_functions_single_test, SparseMatrixTestConfiguration
    benchmark_functions_single_test(SparseMatrixTestConfiguration(1000, 1000, 1., 1000, 1000, 1.), 1)

def multiple_benchmark_example():
    """
    An example of how to benchmark multiple test configurations.
    """
    from src.benchmark import generate_and_benchmark_configurations, show_benchmark_results
    results = generate_and_benchmark_configurations(size_start=100, size_end=1000, size_step=100, density_start=1, density_end=10, density_step=4)
    show_benchmark_results(results)

def main():
    simple_test()
    # single_benchmark_example()
    # multiple_benchmark_example()


if __name__ == '__main__':

    main()