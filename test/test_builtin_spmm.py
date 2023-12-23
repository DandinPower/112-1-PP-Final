import pytest
import torch
from src.utils import generate_sparse_matrix, SparseMatrixTestConfiguration
from src.extension import ExtensionHandler

@pytest.fixture
def handler():
    return ExtensionHandler()

@pytest.mark.parametrize("dim1,dim2,density", [
    (5, 5, 0.7),  # Square matrix
    (10, 10, 0.5),  # Square matrix
    (7, 5, 0.9),  # Rectangular matrix, rows > columns
    (5, 7, 0.8),  # Rectangular matrix, columns > rows
    (6, 6, 0.1),  # Very low density
    (6, 6, 0.9),  # Very high density
    (3, 3, 0.9)  # Original test case
])
def test_sparse_mm(handler, dim1, dim2, density):
    # Generate two sparse matrices
    sparse_matrix1 = generate_sparse_matrix(dim1, dim2, density=density)
    sparse_matrix2 = generate_sparse_matrix(dim2, dim1, density=density)
    test_configuration = SparseMatrixTestConfiguration(dim1, dim2, density, dim2, dim1, density, 1)

    # Multiply the sparse matrices using your function
    result_sparse_true: torch.Tensor = torch.sparse.mm(sparse_matrix1, sparse_matrix2)

@pytest.mark.parametrize("dim1,dim2,dim3,density", [
    (5, 10, 4, 0.7)
])
def test_openmp_sparse_mm(handler, dim1, dim2, dim3, density):
    # Generate two sparse matrices
    sparse_matrix1 = generate_sparse_matrix(dim1, dim2, density=density)
    sparse_matrix2 = generate_sparse_matrix(dim2, dim3, density=density)
    test_configuration = SparseMatrixTestConfiguration(dim1, dim2, density, dim2, dim3, density, 1)

    # Multiply the sparse matrices using your function
    result_sparse_true: torch.Tensor = torch.sparse.mm(sparse_matrix1, sparse_matrix2)