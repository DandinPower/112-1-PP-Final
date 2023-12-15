#pragma once
#include <torch/extension.h>
#include <tuple>
#include <iostream>
#include <omp.h>

#define DEBUG 0
#define N_CPU 16

#define MAX_NNZ_ATOMIC 1


template <typename T, typename T2>
long int omp_get_first(const std::tuple<T, T2>& t) {
    return std::get<0>(t);
}

template <typename T, typename T2>
long int omp_get_first(const at::native::references_holder<T, T2>& t) {
    return at::native::get<0>(t);
}

template <typename scalar_t>
void omp_view(const int64_t n_row, const scalar_t C[])
{
    for (const auto i : c10::irange(n_row))
    {
        std::cout << C[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename scalar_t>
void omp_copy_value(const int64_t n, scalar_t C[], scalar_t D[])
{
    #pragma omp for
    for (int64_t i = 0; i < n; i++)
    {
        C[i] = D[i];
    }
}

torch::Tensor omp_copy_tensor(torch::Tensor tensor)
{
    auto tensor_copy = torch::empty_like(tensor);
    tensor_copy.copy_(tensor);
    return tensor_copy;
}