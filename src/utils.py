from enum import Enum
import dataclasses
import torch
import time

class TestType(Enum):
    BUILTIN = 0
    PARALLEL_STRUCTURE = 1
    OPENMP = 2
    OPENMP_MEM_EFFI = 3
    STD_THREAD = 4

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
    num_threads: int

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

class Logger:
    def __init__(self):
        self.start_time = dict()
        self.end_time = dict()
    
    def reset(self):
        """
        Reset the start and end time dictionaries.
        """
        self.start_time.clear()
        self.end_time.clear()
    
    def start(self, name):
        """
        Start the timer for a given name.

        Parameters:
        name (str): The name of the timer.
        """
        self.start_time[name] = time.perf_counter()
    
    def end(self, name):
        """
        End the timer for a given name.

        Parameters:
        name (str): The name of the timer.
        """
        self.end_time[name] = time.perf_counter()

    def get_total(self):
        """
        Calculate the total time elapsed for all timers.

        Returns:
        float: The total time elapsed in seconds.
        """
        total = 0
        for name in self.start_time:
            total += self.end_time[name] - self.start_time[name]
        return total

    def show(self):
        """
        Print the total time elapsed for all timers in milliseconds.
        """
        total = 0
        for name in self.start_time:
            total += self.end_time[name] - self.start_time[name]
        print(f'Total: {total*1000:.2f}ms')

ConfigNumberType = int | float
def create_config_list_by_mul_step(start: ConfigNumberType, end: ConfigNumberType, step: ConfigNumberType) -> list[ConfigNumberType]:
    config_list = []
    while start <= end:
        config_list.append(start)
        start *= step
    return config_list