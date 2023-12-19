from .utils import generate_sparse_matrix, SparseMatrixTestConfiguration
from src.benchmark_utils import benchmark, builtin_sparse_mm, builtin_sparse_mm_extension, openmp_sparse_mm, std_thread_sparse_mm, dense_mm, BenchmarkResult
from typing import List
from tqdm import tqdm

def _benchmark(config: SparseMatrixTestConfiguration, num_runs: int):
    """
    Benchmark the builtin ``torch.sparse.mm`` function and the extension version of ``torch.sparse.mm`` function.
    Compare it on specific sizes of sparse matrices and specific densities.
    """
    sparse_matrix = generate_sparse_matrix(config.A_row, config.A_col, density=config.A_density)
    sparse_matrix_1 = generate_sparse_matrix(config.B_row, config.B_col, density=config.B_density)

    t0 = 0.0
    t1 = 0.0
    t2 = 0.0
    t3 = 0.0
    t4 = 0.0

    for i in range(num_runs):
        t0 += benchmark(builtin_sparse_mm, sparse_matrix, sparse_matrix_1)
        t1 += benchmark(builtin_sparse_mm_extension, sparse_matrix, sparse_matrix_1)
        t2 += benchmark(openmp_sparse_mm, sparse_matrix, sparse_matrix_1, config.num_threads)
        t3 += benchmark(std_thread_sparse_mm, sparse_matrix, sparse_matrix_1, config.num_threads)
        t4 += benchmark(dense_mm, sparse_matrix.to_dense(), sparse_matrix_1.to_dense())

    return BenchmarkResult(t0 / num_runs, t1 / num_runs, t2 / num_runs, t3 / num_runs, t4 / num_runs, config.A_col, config.A_row, config.A_density, config.B_col, config.B_row, config.B_density, config.num_threads)

def show_benchmark_results(results: List[BenchmarkResult]) -> None:
    """
    Print the benchmark results.
    """
    for result in results:
        print(result)

def benchmark_by_config_list(config_list: List[SparseMatrixTestConfiguration], num_runs: int) -> List[BenchmarkResult]:
    """
    Benchmark the builtin ``torch.sparse.mm`` function and the extension version of ``torch.sparse.mm`` function.
    Compare it on different sizes of sparse matrices and different densities.

    Args:
        config_list (List[SparseMatrixTestConfiguration]): A list of test configurations specifying the sizes and densities of the sparse matrices.
        num_runs (int): The number of times to run the benchmark for each configuration.

    Returns:
        List[BenchmarkResult]: A list of benchmark results, each containing the configuration, average time, and other metrics.
    """
    results = []
    for config in tqdm(config_list):
        results.append(_benchmark(config, num_runs))
    return results

def generate_benchmark_configurations(size_start: int, size_end: int, size_step: int, density_start: float, density_end: float, density_step: float, num_threads_start: int, num_threads_end: int, num_threads_step: int) -> List[SparseMatrixTestConfiguration]:
    """
    Generate a list of benchmark configurations.
    """
    config_list = []
    for size in range(size_start, size_end, size_step):
        for density in range(density_start, density_end, density_step):
            for num_threads in range(num_threads_start, num_threads_end, num_threads_step):
                config_list.append(SparseMatrixTestConfiguration(size, size, density / 10, size, size, density / 10, num_threads))
    return config_list