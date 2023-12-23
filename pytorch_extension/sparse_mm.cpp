#include <torch/extension.h>
#include <iostream>
#include <builtin.h>
#include <parallel_structure.h>
#include <omp_mm.h>
#include <omp_mem_efficient.h>
#include <std_thread.h>
#include <logger.h>

// if you want to show the logs, you can set VERBOSE to 1
int VERBOSE = 0;

void set_verbose(int verbose)
{
    VERBOSE = verbose;
}

torch::Tensor sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1, TestConfig config)
{
    logger.reset();
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = builtin::sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1);
    // logger.showLogs(config);
    // because it is hard to compare with the parallel structure code, so we compare the builtin implementation by using parallel structure refactoring code
    return answer;
}

torch::Tensor parallel_structure_sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1, TestConfig config)
{
    logger.reset();
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = parallel_structure::sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1);
    if (VERBOSE)
        logger.showLogs(config);
    return answer;
}

torch::Tensor openmp_sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1, const int num_threads, TestConfig config)
{
    logger.reset();
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = omp::sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1, num_threads);

    if (VERBOSE)
        logger.showLogs(config);
    return answer;
}

torch::Tensor openmp_mem_efficient_sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1, const int num_threads, TestConfig config)
{
    logger.reset();
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = omp_mem_efficient::sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1, num_threads);

    // different from the parallel structure code, we don't need to show the logs here
    // logger.showLogs(config);
    return answer;
}

torch::Tensor std_thread_sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1, const int num_threads, TestConfig config)
{
    logger.reset();
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = std_thread::sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1, num_threads);
    
    if (VERBOSE)
        logger.showLogs(config);
    return answer;
}

/**
 * @brief Binds the module to the Python interpreter.
 *
 * This function is used to bind the module to the Python interpreter using Pybind11.
 * It defines the module name and exposes the necessary functions and classes to Python.
 *
 * @param m The module object representing the module being bound.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<TestType>(m, "TestType")
        .value("BUILTIN", TestType::BUILTIN)
        .value("PARALLEL_STRUCTURE", TestType::PARALLEL_STRUCTURE)
        .value("OPENMP", TestType::OPENMP)
        .value("OPENMP_MEM_EFFI", TestType::OPENMP_MEM_EFFI)
        .value("STD_THREAD", TestType::STD_THREAD)
        .export_values();

    py::class_<TestConfig>(m, "TestConfig")
        .def(py::init<>())
        .def_readwrite("test_type", &TestConfig::test_type)
        .def_readwrite("A_row", &TestConfig::A_row)
        .def_readwrite("A_col", &TestConfig::A_col)
        .def_readwrite("A_density", &TestConfig::A_density)
        .def_readwrite("B_row", &TestConfig::B_row)
        .def_readwrite("B_col", &TestConfig::B_col)
        .def_readwrite("B_density", &TestConfig::B_density)
        .def_readwrite("num_threads", &TestConfig::num_threads);
    m.def("sparse_mm", &sparse_mm, "Sparse matrix multiplication");
    m.def("parallel_structure_sparse_mm", &parallel_structure_sparse_mm, "Parallel Structure Refactorization Sparse matrix multiplication");
    m.def("openmp_sparse_mm", &openmp_sparse_mm, "OpenMP Sparse matrix multiplication");
    m.def("openmp_mem_efficient_sparse_mm", &openmp_mem_efficient_sparse_mm, "OpenMP Mem Efficient Sparse matrix multiplication");
    m.def("std_thread_sparse_mm", &std_thread_sparse_mm, "std::thread Sparse matrix multiplication");
    m.def("set_verbose", &set_verbose, "set verbose");
}