import torch
import sparse_mm
import multiprocessing

num_cores = multiprocessing.cpu_count()

class ExtensionHandler(object):
    @staticmethod
    def sparse_mm(sparse_matrix, sparse_matrix_1):
        """
        Computes the product of two sparse matrices by using the extension version of ``torch.sparse.mm`` function
        """
        return sparse_mm.sparse_mm(sparse_matrix, sparse_matrix_1)
    
    @staticmethod
    def openmp_sparse_mm(sparse_matrix, sparse_matrix_1, num_threads=num_cores):
        """
        Computes the product of two sparse matrices by using the openmp parallel version of ``torch.sparse.mm`` function
        """
        return sparse_mm.openmp_sparse_mm(sparse_matrix, sparse_matrix_1, num_threads)
    
    @staticmethod
    def std_thread_sparse_mm(sparse_matrix, sparse_matrix_1, num_threads=num_cores):
        """
        Computes the product of two sparse matrices by using the std::thread parallel version of ``torch.sparse.mm`` function
        """
        return sparse_mm.std_thread_sparse_mm(sparse_matrix, sparse_matrix_1, num_threads)