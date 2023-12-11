import torch.utils.benchmark as benchmark
from .utils import generate_sparse_matrix, SparseMatrixTestConfiguration
from typing import List

# WARNING: Do not use the following function for benchmarking. Because when using benchmark.Timer
# it doesn't utilize all core to run c++ part. so the result is not accurate.

def benchmark_functions_single_test(config: SparseMatrixTestConfiguration, num_runs):
    """
    Benchmark the builtin ``torch.sparse.mm`` function and the extension version of ``torch.sparse.mm`` function.
    Compare it on specific sizes of sparse matrices and specific densities.
    """
    sparse_matrix = generate_sparse_matrix(config.A_row, config.A_col, density=config.A_density)
    sparse_matrix_1 = generate_sparse_matrix(config.B_row, config.B_col, density=config.B_density)

    t0 = benchmark.Timer(
        stmt='builtin_sparse_mm(sparse_matrix, sparse_matrix_1)',
        setup='from src.benchmark_utils import builtin_sparse_mm',
        globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1})

    t1 = benchmark.Timer(
        stmt='builtin_sparse_mm_extension(sparse_matrix, sparse_matrix_1)',
        setup='from src.benchmark_utils import builtin_sparse_mm_extension',
        globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1})
    
    t2 = benchmark.Timer(
        stmt='openmp_sparse_mm(sparse_matrix, sparse_matrix_1)',
        setup='from src.benchmark_utils import openmp_sparse_mm',
        globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1})

    t3 = benchmark.Timer(
        stmt='dense_mm(sparse_matrix.to_dense(), sparse_matrix_1.to_dense())',
        setup='from src.benchmark_utils import dense_mm',
        globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1})

    print(f'builtin_sparse_mm(sparse_matrix, sparse_matrix_1):  {t0.timeit(num_runs)})')
    print(f'builtin_sparse_mm_extension(sparse_matrix, sparse_matrix_1):      {t1.timeit(num_runs)})')
    print(f'openmp_sparse_mm(sparse_matrix, sparse_matrix_1):      {t2.timeit(num_runs)})')
    print(f'dense_mm(sparse_matrix.to_dense(), sparse_matrix_1.to_dense()):      {t2.timeit(num_runs)})')
    print(f'Finsish Benchmarking with A_row={config.A_row}, A_col={config.A_col}, A_density={config.A_density}, B_row={config.B_row}, B_col={config.B_col}, B_density={config.B_density}, num_runs={num_runs}')
    print("-" * 50 + "\n")

# WARNING: This function may cause CUDA kernel timeout errors if the benchmark function call time exceeds a certain threshold.
# SOLUTION: Limit the number of configurations in the list to prevent this issue.
def benchmark_functions_multiple_test(config_list: List[SparseMatrixTestConfiguration], results: List):
    """
    Benchmark the builtin ``torch.sparse.mm`` function and the extension version of ``torch.sparse.mm`` function.
    Compare it on different sizes of sparse matrices and different densities.
    """

    for i, config in enumerate(config_list):
        label = 'Benchmarking SPMM strategies'
        sub_label = f'[{config.A_row:5}, {config.A_col:5}, density={config.A_density:5.2f}], [{config.B_row:5}, {config.B_col:5}, density={config.B_density:5.2f}]'
        sparse_matrix = generate_sparse_matrix(config.A_row, config.A_col, density=config.A_density)
        sparse_matrix_1 = generate_sparse_matrix(config.B_row, config.B_col, density=config.B_density)
        # print(f'Builtin SPMM: {i*4 + 0}')
        results.append(benchmark.Timer(
            stmt='builtin_sparse_mm(sparse_matrix, sparse_matrix_1)',
            setup='from src.benchmark_utils import builtin_sparse_mm',
            globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1},
            label=label,
            sub_label=sub_label,
            description='builtin_sparse_mm').timeit(1))
        
        # print(f'Extension SPMM: {i*4 + 1}')
        results.append(benchmark.Timer(
            stmt='builtin_sparse_mm_extension(sparse_matrix, sparse_matrix_1)',
            setup='from src.benchmark_utils import builtin_sparse_mm_extension',
            globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1},
            label=label,
            sub_label=sub_label,
            description='builtin_sparse_mm_extension').timeit(1))
        
        # print(f'Parallel SPMM: {i*4 + 2}')
        results.append(benchmark.Timer(
            stmt='openmp_sparse_mm(sparse_matrix, sparse_matrix_1)',
            setup='from src.benchmark_utils import openmp_sparse_mm',
            globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1},
            label=label,
            sub_label=sub_label,
            description='openmp_sparse_mm').timeit(1))

        # print(f'Dense MM: {i*4 + 2}')
        results.append(benchmark.Timer(
            stmt='dense_mm(sparse_matrix.to_dense(), sparse_matrix_1.to_dense())',
            setup='from src.benchmark_utils import dense_mm',
            globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1},
            label=label,
            sub_label=sub_label,
            description='dense_mm').timeit(1))
        
def show_benchmark_results(results):
    """
    Show the benchmark results.
    """
    compare = benchmark.Compare(results)
    compare.print()

def generate_and_benchmark_configurations(size_start, size_end, size_step, density_start, density_end, density_step):
    """
    Generate a list of test configurations and benchmark them.
    """
    size_range = range(size_start, size_end, size_step)
    density_range = range(density_start, density_end, density_step)
    results = []
    from tqdm import tqdm
    for size in tqdm(size_range, desc="Size"):
        configurations = []
        for density in [i/10 for i in density_range]:
            for density_2 in [i/10 for i in density_range]:
                configurations.append(SparseMatrixTestConfiguration(size, size, density, size, size, density_2))
        benchmark_functions_multiple_test(configurations, results)
    return results