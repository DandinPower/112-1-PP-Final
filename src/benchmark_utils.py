import torch
import time
import dataclasses
from .extension import ExtensionHandler

def benchmark(func, *args):
    """Benchmark the execution time of a function"""
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

def builtin_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two sparse matrices by using the builtin ``torch.sparse.mm`` function"""
    return torch.sparse.mm(sparse_matrix, sparse_matrix_1)

def builtin_sparse_mm_extension(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two sparse matrices by using the extension version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.sparse_mm(sparse_matrix, sparse_matrix_1)

def openmp_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two sparse matrices by using the openmp version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.openmp_sparse_mm(sparse_matrix, sparse_matrix_1)

def std_thread_sparse_mm(sparse_matrix: torch.Tensor, sparse_matrix_1: torch.Tensor):
    """Computes the product of two sparse matrices by using the pthread version of ``torch.sparse.mm`` function"""
    return ExtensionHandler.std_thread_sparse_mm(sparse_matrix, sparse_matrix_1)

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
    openmp_sparse_mm: float
    std_thread_sparse_mm: float
    dense_mm: float
    A_col: int
    A_row: int
    A_density: float
    B_col: int
    B_row: int
    B_density: float

    def __repr__(self) -> str:
        """Return the string representation of the benchmark result"""
        return_string = f'A=[ {self.A_row} x {self.A_col}, {self.A_density}], B=[ {self.B_row} x {self.B_col}, {self.B_density} ]\n'
        return_string += "-" * 50 + "\n"
        return_string += f'builtin_sparse_mm: {self.sparse_mm:>20.5f}ms\n'
        return_string += f'builtin_sparse_mm_extension: {self.sparse_mm_extension:>10.5f}ms\n'
        return_string += f'openmp_sparse_mm: {self.openmp_sparse_mm:>21.5f}ms\n'
        return_string += f'std_thread_sparse_mm: {self.std_thread_sparse_mm:>17.5f}ms\n'
        return_string += f'dense_mm: {self.dense_mm:>29.5f}ms\n'
        return_string += "-" * 50 + "\n"
        return return_string