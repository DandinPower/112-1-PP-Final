#include <torch/extension.h>
#include <iostream>
#include <omp.h>
#include "include/builtin.h"
#include "include/omp_mm.h"

torch::Tensor sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1)
{
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1);
    return answer;
}

torch::Tensor openmp_sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1)
{
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    omp_set_num_threads(N_CPU);
    
    // TODO: Implement OpenMP sparse matrix multiplication to replace the original sparse matrix multiplication
    torch::Tensor answer = sparse_sparse_matmul_omp(sparse_matrix_0, sparse_matrix_1);
    return answer;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_mm", &sparse_mm, "Sparse matrix multiplication");
    m.def("openmp_sparse_mm", &openmp_sparse_mm, "OpenMP Sparse matrix multiplication");
}