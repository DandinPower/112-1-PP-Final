import torch

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
    # Generate a dense matrix with random values
    dense_matrix = torch.rand(rows, cols)

    # Create a mask for the non-zero elements
    mask = torch.rand(rows, cols) < density

    # Apply the mask to the dense matrix
    dense_matrix = dense_matrix * mask

    # Convert the dense matrix to a sparse matrix
    sparse_matrix = dense_matrix.to_sparse()

    return sparse_matrix