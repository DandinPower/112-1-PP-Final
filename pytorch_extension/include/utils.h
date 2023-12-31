#pragma once
#include <torch/extension.h>
#include <tuple>
#include <iostream>

template <typename T, typename T2>
long int get_first(const std::tuple<T, T2>& t) {
    return std::get<0>(t);
}

template <typename T, typename T2>
long int get_first(const at::native::references_holder<T, T2>& t) {
    return at::native::get<0>(t);
}

template <typename scalar_t>
void view(const int64_t n_row, const scalar_t C[])
{
    for (const auto i : c10::irange(n_row))
    {
        std::cout << C[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename scalar_t>
void copy_value(const int64_t n, scalar_t C[], scalar_t D[])
{
    for (const auto i : c10::irange(n))
    {
        C[i] = D[i];
    }
}

torch::Tensor copy_tensor(torch::Tensor tensor)
{
    auto tensor_copy = torch::empty_like(tensor);
    tensor_copy.copy_(tensor);
    return tensor_copy;
}