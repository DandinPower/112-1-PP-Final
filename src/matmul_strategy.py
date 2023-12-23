from src.utils import SparseMatrixTestConfiguration
import torch

class MatmulStrategy:
    def __init__(self):
        self.fake_config = SparseMatrixTestConfiguration(0, 0, 0, 0, 0, 0, 0)
        self.matmul_strategy = None
        self.thread_num = -1

    def set_matmul_strategy(self, matmul_strategy):
        self.matmul_strategy = matmul_strategy

    def set_thread_num(self, num: int) -> None:
        self.thread_num = num

    def matmul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert self.matmul_strategy is not None, 'Please set matmul strategy first.'
        # strategy don't need to know the thread num
        if self.thread_num != -1:
            return self.matmul_strategy(x, y, self.fake_config, self.thread_num)
        return self.matmul_strategy(x, y, self.fake_config)