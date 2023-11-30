import torch
import sparse_mm

# Create some sparse and dense tensors
sparse = torch.sparse_coo_tensor(
    indices=torch.tensor([[0, 1, 1],
                          [2, 0, 2]]),
    values=torch.tensor([3., 4., 5.]),
    size=(2, 3))

dense = torch.randn(3, 2)

# Use the extension for sparse matrix multiplication
result = sparse_mm.sparse_mm(sparse, dense)

print(result)