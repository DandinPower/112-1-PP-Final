import pytest
import torch
from src.utils import generate_sparse_matrix
from src.extension import ExtensionHandler

@pytest.fixture
def handler():
    return ExtensionHandler()

@pytest.mark.parametrize("dim1,dim2,density", [(5, 4, 0.7), (10, 10, 0.5), (3, 3, 0.9)])
def test_pthread_sparse_mm(handler, dim1, dim2, density):
    # Generate two sparse matrices
    sparse_matrix1 = generate_sparse_matrix(dim1, dim2, density=density)
    sparse_matrix2 = generate_sparse_matrix(dim2, dim1, density=density)

    # Multiply the sparse matrices using your function
    result_sparse: torch.Tensor = handler.std_thread_sparse_mm(sparse_matrix1, sparse_matrix2)
    result_sparse_true: torch.Tensor = torch.sparse.mm(sparse_matrix1, sparse_matrix2)

    # Check if the result from your function matches the expected result
    assert torch.allclose(result_sparse.to_dense(), result_sparse_true.to_dense())