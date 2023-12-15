#include <torch/extension.h>
#include <iostream>
#include <builtin.h>
#include <builtin_omp.h>
#include <builtin_std_thread.h>

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

    torch::Tensor answer = sparse_sparse_matmul_cpu_omp(sparse_matrix_0, sparse_matrix_1);
    return answer;
}

torch::Tensor std_thread_sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1)
{
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    // torch::Tensor answer = sparse_sparse_matmul_cpu_std_thread(sparse_matrix_0, sparse_matrix_1);
    torch::Tensor answer = sparse_sparse_matmul_cpu_std_thread(sparse_matrix_0, sparse_matrix_1);
    return answer;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_mm", &sparse_mm, "Sparse matrix multiplication");
    m.def("openmp_sparse_mm", &openmp_sparse_mm, "OpenMP Sparse matrix multiplication");
    m.def("std_thread_sparse_mm", &std_thread_sparse_mm, "std::thread Sparse matrix multiplication");
}