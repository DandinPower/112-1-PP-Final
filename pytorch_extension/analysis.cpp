#include <iostream>
#include <torch/extension.h>
#include "include/builtin.h"


torch::Tensor sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1)
{
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");

    torch::Tensor answer = sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1);
    return answer;
}


int main(){

    c10::TensorOptions opts(c10::ScalarType::Float);

    torch::Tensor a = torch::rand({1000, 1000}, opts).to_sparse();
    torch::Tensor b = torch::rand({1000, 1000}, opts).to_sparse();

    torch::Tensor c = sparse_mm(a, b);

    return 0;
}