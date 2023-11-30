import torch

# Define two dense matrices
dense1 = torch.tensor([[0., 0., 3.], [4., 0., 5.]])
dense2 = torch.tensor([[0., 2.], [0., 0.], [3., 0.]])

# Convert the dense matrices to sparse
sparse1 = dense1.to_sparse()
sparse2 = dense2.to_sparse()

# Multiply the sparse matrices
result = torch.sparse.mm(sparse1, sparse2)

print("Result of multiplication:")
print(result.to_dense())