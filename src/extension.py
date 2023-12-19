import torch
import sparse_mm
import multiprocessing
from src.utils import SparseMatrixTestConfiguration, TestType

num_cores = multiprocessing.cpu_count()

class ExtensionHandler(object):
    @staticmethod
    def sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration: SparseMatrixTestConfiguration):
        """
        Computes the product of two sparse matrices by using the extension version of ``torch.sparse.mm`` function
        """
        extension_test_configuration = ExtensionHandler.get_extension_config(test_configuration)
        ExtensionHandler.set_extension_config_type(extension_test_configuration, TestType.BUILTIN)
        return sparse_mm.sparse_mm(sparse_matrix, sparse_matrix_1, extension_test_configuration)
    
    @staticmethod
    def parallel_structure_sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration: SparseMatrixTestConfiguration):
        """
        Computes the product of two sparse matrices by using the parallel structure version (without actually parallelism) of ``torch.sparse.mm`` function
        """
        extension_test_configuration = ExtensionHandler.get_extension_config(test_configuration)
        ExtensionHandler.set_extension_config_type(extension_test_configuration, TestType.PARALLEL_STRUCTURE)
        return sparse_mm.parallel_structure_sparse_mm(sparse_matrix, sparse_matrix_1, extension_test_configuration)
    
    @staticmethod
    def openmp_sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration: SparseMatrixTestConfiguration, num_threads=num_cores):
        """
        Computes the product of two sparse matrices by using the openmp parallel version of ``torch.sparse.mm`` function
        """
        extension_test_configuration = ExtensionHandler.get_extension_config(test_configuration)
        ExtensionHandler.set_extension_config_type(extension_test_configuration, TestType.OPENMP)
        return sparse_mm.openmp_sparse_mm(sparse_matrix, sparse_matrix_1, num_threads, extension_test_configuration)
    
    @staticmethod
    def std_thread_sparse_mm(sparse_matrix, sparse_matrix_1, test_configuration: SparseMatrixTestConfiguration, num_threads=num_cores):
        """
        Computes the product of two sparse matrices by using the std::thread parallel version of ``torch.sparse.mm`` function
        """
        extension_test_configuration = ExtensionHandler.get_extension_config(test_configuration)
        ExtensionHandler.set_extension_config_type(extension_test_configuration, TestType.STD_THREAD)
        return sparse_mm.std_thread_sparse_mm(sparse_matrix, sparse_matrix_1, num_threads, extension_test_configuration)
    
    @staticmethod
    def get_extension_config(test_configuration: SparseMatrixTestConfiguration):
        """
        Converts a SparseMatrixTestConfiguration object into an extension_test_configuration object.

        Args:
        - test_configuration: SparseMatrixTestConfiguration object to be converted.

        Returns:
        - extension_test_configuration: Converted extension_test_configuration object.
        """
        extension_test_configuration = sparse_mm.TestConfig()
        extension_test_configuration.A_col = test_configuration.A_col
        extension_test_configuration.A_row = test_configuration.A_row
        extension_test_configuration.A_density = test_configuration.A_density
        extension_test_configuration.B_col = test_configuration.B_col
        extension_test_configuration.B_row = test_configuration.B_row
        extension_test_configuration.B_density = test_configuration.B_density
        extension_test_configuration.num_threads = test_configuration.num_threads
        return extension_test_configuration
    
    @staticmethod
    def set_extension_config_type(extension_test_configuration, test_type):
        """
        Set the test type for the extension test configuration.

        Args:
        - extension_test_configuration: The extension test configuration object.
        - test_type: The test type to be set.

        Raises:
        - ValueError: If the test type is invalid.
        """
        if test_type == TestType.BUILTIN:
            extension_test_configuration.test_type = sparse_mm.TestType.BUILTIN
        elif test_type == TestType.PARALLEL_STRUCTURE:
            extension_test_configuration.test_type = sparse_mm.TestType.PARALLEL_STRUCTURE
        elif test_type == TestType.OPENMP:
            extension_test_configuration.test_type = sparse_mm.TestType.OPENMP
        elif test_type == TestType.STD_THREAD:
            extension_test_configuration.test_type = sparse_mm.TestType.STD_THREAD
        else:
            raise ValueError('Invalid test type')
