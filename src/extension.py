import torch
import sparse_mm

class ExtensionHandler(object):
    @staticmethod
    def sparse_mm(sparse_matrix, sparse_matrix_1):
        """
        Computes the product of two sparse matrices by using the extension version of ``torch.sparse.mm`` function
        """
        return sparse_mm.sparse_mm(sparse_matrix, sparse_matrix_1)