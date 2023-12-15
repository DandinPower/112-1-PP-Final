import torch
from torch.utils.benchmark import Measurement

import dataclasses
import matplotlib.pyplot as plt
from typing import List


plt.style.use("seaborn")


@dataclasses.dataclass
class SparseMatrixTestConfiguration(object):
    """
    A test configuration for sparse matrix multiplication.
    """
    A_row: int
    A_col: int
    A_density: float
    B_row: int
    B_col: int
    B_density: float    

def generate_sparse_matrix(rows, cols, density=0.1):
    """
    Generate a random sparse matrix.

    Parameters:
    rows (int): The number of rows in the matrix.
    cols (int): The number of columns in the matrix.
    density (float): The density of the sparse matrix. Default is 0.1.

    Returns:
    torch.Tensor: The generated sparse matrix.
    """
    dense_matrix = torch.rand(rows, cols)
    mask = torch.rand(rows, cols) < density
    dense_matrix = dense_matrix * mask
    sparse_matrix = dense_matrix.to_sparse().to(device='cpu')
    return sparse_matrix

def assert_sparse_or_dense_matrix_are_equal(matrix: torch.Tensor, matrix_1: torch.Tensor):
    """
    Assert that two matrices are equal, regardless of whether they are sparse or dense.
    """
    if matrix.is_sparse:
        matrix = matrix.to_dense()
    if matrix_1.is_sparse:
        matrix_1 = matrix_1.to_dense()
    assert matrix.allclose(matrix_1)
    return True

def plot_results(results: List[Measurement]):
    exe_times_dict = {
        "builtin_sparse_mm"           : [],
        "builtin_sparse_mm_extension" : [], 
        "openmp_sparse_mm"            : [],
        "dense_mm"                    : []
    }

    for res in results:
        exe_times_dict[res.task_spec.stmt.split("(")[0]].append(res.raw_times)
    
    plt.plot(exe_times_dict["builtin_sparse_mm"])
    plt.plot(exe_times_dict["builtin_sparse_mm_extension"])
    plt.plot(exe_times_dict["openmp_sparse_mm"])
    plt.plot(exe_times_dict["dense_mm"])
    plt.legend(list(exe_times_dict.keys()))
    plt.title("Execution Times")
    plt.savefig("plot/Test.png")