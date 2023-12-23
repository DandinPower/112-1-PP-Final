import torch
import time
import dataclasses
from src.utils import SparseMatrixTestConfiguration
from .extension import ExtensionHandler

def benchmark(func, *args):
    """Benchmark the execution time of a function"""
    start_time = time.perf_counter()
    func(*args)
    end_time = time.perf_counter()
    return end_time - start_time

def builtin_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two sparse matrices by using the builtin ``torch.sparse.mm`` function"""
    return torch.sparse.mm(sparse_matrix, sparse_matrix_1)

def builtin_sparse_mm_extension(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor, test_configuration: SparseMatrixTestConfiguration):
    """Computes the product of two sparse matrices by using the extension version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration)

def parallel_structure_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor, test_configuration: SparseMatrixTestConfiguration):
    """Computes the product of two sparse matrices by using the parallel structure version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.parallel_structure_sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration)

def openmp_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor, num_threads: int, test_configuration: SparseMatrixTestConfiguration):
    """Computes the product of two sparse matrices by using the openmp version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.openmp_sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration, num_threads)

def openmp_mem_effi_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor, num_threads: int, test_configuration: SparseMatrixTestConfiguration):
    """Computes the product of two sparse matrices by using the openmp version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.openmp_mem_effi_sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration, num_threads)

def std_thread_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor, num_threads: int, test_configuration: SparseMatrixTestConfiguration):
    """Computes the product of two sparse matrices by using the pthread version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.std_thread_sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration, num_threads)

def dense_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two dense matrices by using the builtin ``torch.mm`` function"""
    if sparse_matrix.is_sparse:
        sparse_matrix = sparse_matrix.to_dense()
    if sparse_matrix_1.is_sparse:
        sparse_matrix_1 = sparse_matrix_1.to_dense()
    return torch.mm(sparse_matrix, sparse_matrix_1)

@dataclasses.dataclass
class BenchmarkResult:
    """A dataclass that stores the benchmark result"""
    sparse_mm: float
    sparse_mm_extension: float
    parallel_structure_sparse_mm: float
    openmp_sparse_mm: float
    openmp_mem_effi_sparse_mm: float
    std_thread_sparse_mm: float
    dense_mm: float
    A_col: int
    A_row: int
    A_density: float
    B_col: int
    B_row: int
    B_density: float
    num_threads: int

    def __repr__(self) -> str:
        """Return the string representation of the benchmark result"""
        return_string = f'A=[ {self.A_row} x {self.A_col}, {self.A_density}], B=[ {self.B_row} x {self.B_col}, {self.B_density}], num_threads= [ {self.num_threads} ]\n'
        return_string += "-" * 50 + "\n"
        return_string += f'builtin_sparse_mm: {self.sparse_mm*1000:>25.5f}ms\n'
        return_string += f'builtin_sparse_mm_extension: {self.sparse_mm_extension*1000:>15.5f}ms\n'
        return_string += f'parallel_structure_sparse_mm: {self.parallel_structure_sparse_mm*1000:>14.5f}ms\n'
        return_string += f'openmp_sparse_mm: {self.openmp_sparse_mm*1000:>26.5f}ms\n'
        return_string += f'openmp_mem_effi_sparse_mm: {self.openmp_mem_effi_sparse_mm*1000:>17.5f}ms\n'
        return_string += f'std_thread_sparse_mm: {self.std_thread_sparse_mm*1000:>22.5f}ms\n'
        return_string += f'dense_mm: {self.dense_mm*1000:>34.5f}ms\n'
        return_string += "-" * 50 + "\n"
        return return_string