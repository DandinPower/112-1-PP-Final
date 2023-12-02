import unittest
import torch
from src.utils import generate_sparse_matrix
from src.extension import ExtensionHandler

class TestSparseMatrixMultiplication(unittest.TestCase):
    def test_sparse_mm(self):
        # Generate two sparse matrices
        sparse_matrix1 = generate_sparse_matrix(5, 4, density=0.7)
        sparse_matrix2 = generate_sparse_matrix(4, 5, density=0.7)

        # Multiply the sparse matrices using your function
        result_sparse = ExtensionHandler.sparse_mm(sparse_matrix1, sparse_matrix2)
        result_sparse_true = torch.sparse.mm(sparse_matrix1, sparse_matrix2)
        print(sparse_matrix1.to_dense())
        print(result_sparse.to_dense())
        print(result_sparse_true.to_dense())

        # Check if the result from your function matches the expected result
        self.assertTrue(torch.allclose(result_sparse.to_dense(), result_sparse_true.to_dense()))

if __name__ == "__main__":
    unittest.main()