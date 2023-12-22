import multiprocessing
import pytest
import torch
from src.utils import generate_sparse_matrix, SparseMatrixTestConfiguration
from src.extension import ExtensionHandler

num_cores = multiprocessing.cpu_count()

@pytest.fixture
def handler():
    return ExtensionHandler()

@pytest.mark.parametrize("dim1,dim2,density,num_threads", [
    (5, 5, 0.7, 1),  # Single thread
    (10, 10, 0.5, 2),  # Multiple threads
    (7, 5, 0.9, 4),  # Multiple threads
    (5, 7, 0.8, 8),  # Multiple threads
    (6, 6, 0.1, num_cores),  # Maximum threads
    (6, 6, 0.9, 1),  # Single thread
    (3, 3, 0.9, 2)  # Multiple threads
])
def test_omp_sparse_mm(handler, dim1, dim2, density, num_threads):
    # Generate two sparse matrices
    sparse_matrix1 = generate_sparse_matrix(dim1, dim2, density=density)
    sparse_matrix2 = generate_sparse_matrix(dim2, dim1, density=density)
    test_configuration = SparseMatrixTestConfiguration(dim1, dim2, density, dim2, dim1, density, num_threads)

    # Multiply the sparse matrices using your function
    result_sparse: torch.Tensor = handler.openmp_sparse_mm(sparse_matrix1, sparse_matrix2, test_configuration, num_threads=num_threads)
    result_sparse_true: torch.Tensor = torch.sparse.mm(sparse_matrix1, sparse_matrix2)

    # Check if the result from your function matches the expected result
    assert torch.allclose(result_sparse.to_dense(), result_sparse_true.to_dense())

# @pytest.mark.parametrize("dim1,dim2,dim3,density,num_thread", [
#     (5, 10, 4, 0.7, 1)
# ])
# def test_openmp_sparse_mm(handler, dim1, dim2, dim3, density, num_thread):
#     # Generate two sparse matrices
#     sparse_matrix1 = generate_sparse_matrix(dim1, dim2, density=density)
#     sparse_matrix2 = generate_sparse_matrix(dim2, dim3, density=density)
#     test_configuration = SparseMatrixTestConfiguration(dim1, dim2, density, dim2, dim3, density, 1)

#     # Multiply the sparse matrices using your function
#     result_sparse: torch.Tensor = handler.openmp_sparse_mm(sparse_matrix1, sparse_matrix2, test_configuration, num_threads=num_thread)
#     result_sparse_true: torch.Tensor = torch.sparse.mm(sparse_matrix1, sparse_matrix2)

#     # Check if the result from your function matches the expected result
#     assert torch.allclose(result_sparse.to_dense(), result_sparse_true.to_dense())