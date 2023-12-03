import torch
import torch.utils.benchmark as benchmark
from itertools import product
from .extension import ExtensionHandler
from .utils import generate_sparse_matrix, SparseMatrixTestConfiguration
from typing import List

def builtin_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two sparse matrices by using the builtin ``torch.sparse.mm`` function"""
    return torch.sparse.mm(sparse_matrix, sparse_matrix_1)

def builtin_sparse_mm_extension(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two sparse matrices by using the extension version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.sparse_mm(sparse_matrix, sparse_matrix_1)

def dense_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two dense matrices by using the builtin ``torch.mm`` function"""
    if sparse_matrix.is_sparse:
        sparse_matrix = sparse_matrix.to_dense()
    if sparse_matrix_1.is_sparse:
        sparse_matrix_1 = sparse_matrix_1.to_dense()
    return torch.mm(sparse_matrix, sparse_matrix_1)

def benchmark_functions_single_test(config: SparseMatrixTestConfiguration, num_runs):
    """
    Benchmark the builtin ``torch.sparse.mm`` function and the extension version of ``torch.sparse.mm`` function.
    Compare it on specific sizes of sparse matrices and specific densities.
    """
    sparse_matrix = generate_sparse_matrix(config.A_row, config.A_col, density=config.A_density)
    sparse_matrix_1 = generate_sparse_matrix(config.B_row, config.B_col, density=config.B_density)

    t0 = benchmark.Timer(
        stmt='builtin_sparse_mm(sparse_matrix, sparse_matrix_1)',
        setup='from src.benchmark import builtin_sparse_mm',
        globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1})

    t1 = benchmark.Timer(
        stmt='builtin_sparse_mm_extension(sparse_matrix, sparse_matrix_1)',
        setup='from src.benchmark import builtin_sparse_mm_extension',
        globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1})

    t2 = benchmark.Timer(
        stmt='dense_mm(sparse_matrix.to_dense(), sparse_matrix_1.to_dense())',
        setup='from src.benchmark import dense_mm',
        globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1})

    print(f'builtin_sparse_mm(sparse_matrix, sparse_matrix_1):  {t0.timeit(num_runs)})')
    print(f'builtin_sparse_mm_extension(sparse_matrix, sparse_matrix_1):      {t1.timeit(num_runs)})')
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
        print(f'Builtin SPMM: {i*3 + 0}')
        results.append(benchmark.Timer(
            stmt='builtin_sparse_mm(sparse_matrix, sparse_matrix_1)',
            setup='from src.benchmark import builtin_sparse_mm',
            globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1},
            label=label,
            sub_label=sub_label,
            description='builtin_sparse_mm').timeit(1))
        
        print(f'Extension SPMM: {i*3 + 1}')
        results.append(benchmark.Timer(
            stmt='builtin_sparse_mm_extension(sparse_matrix, sparse_matrix_1)',
            setup='from src.benchmark import builtin_sparse_mm_extension',
            globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1},
            label=label,
            sub_label=sub_label,
            description='builtin_sparse_mm_extension').timeit(1))

        print(f'Dense MM: {i*3 + 2}')
        results.append(benchmark.Timer(
            stmt='dense_mm(sparse_matrix.to_dense(), sparse_matrix_1.to_dense())',
            setup='from src.benchmark import dense_mm',
            globals={'sparse_matrix': sparse_matrix, 'sparse_matrix_1': sparse_matrix_1},
            label=label,
            sub_label=sub_label,
            description='dense_mm').timeit(1))
        
def show_benchmark_results(results):
    """
    Show the benchmark results.
    """
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()
