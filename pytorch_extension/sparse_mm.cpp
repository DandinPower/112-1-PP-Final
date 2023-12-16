#include <torch/extension.h>
#include <iostream>
#include <thread>
#include <builtin.h>
#include <builtin_omp.h>
#include <builtin_std_thread.h>

const unsigned int THREADS = std::thread::hardware_concurrency();

/**
 * Performs sparse matrix multiplication using two sparse tensors.
 *
 * @param sparse_matrix_0 The first sparse tensor.
 * @param sparse_matrix_1 The second sparse tensor.
 * @return The result of the sparse matrix multiplication.
 */
torch::Tensor sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1)
{
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1);
    return answer;
}

/**
 * Performs sparse matrix multiplication using OpenMP.
 * 
 * @param sparse_matrix_0 The first sparse matrix tensor.
 * @param sparse_matrix_1 The second sparse matrix tensor.
 * @param num_threads The number of threads to use for parallel computation.
 * @return The result of the sparse matrix multiplication.
 */
torch::Tensor openmp_sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1, const int num_threads)
{
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = sparse_sparse_matmul_cpu_omp(sparse_matrix_0, sparse_matrix_1, num_threads);
    return answer;
}

/**
 * Performs sparse matrix multiplication using standard threads.
 * 
 * @param sparse_matrix_0 The first sparse matrix tensor.
 * @param sparse_matrix_1 The second sparse matrix tensor.
 * @param num_threads The number of threads to use for the computation.
 * @return The result of the sparse matrix multiplication.
 */
torch::Tensor std_thread_sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1, const int num_threads)
{
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = sparse_sparse_matmul_cpu_std_thread(sparse_matrix_0, sparse_matrix_1, num_threads);
    return answer;
}

/**
 * @brief Binds the functions to the Python module.
 *
 * This function is used to bind the functions `sparse_mm`, `openmp_sparse_mm`, and `std_thread_sparse_mm` to the Python module.
 *
 * @param m The Python module to bind the functions to.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_mm", &sparse_mm, "Sparse matrix multiplication");
    m.def("openmp_sparse_mm", &openmp_sparse_mm, "OpenMP Sparse matrix multiplication");
    m.def("std_thread_sparse_mm", &std_thread_sparse_mm, "std::thread Sparse matrix multiplication");
}