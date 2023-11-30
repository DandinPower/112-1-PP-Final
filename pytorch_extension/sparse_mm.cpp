#include <torch/extension.h>
#include <iostream>

torch::Tensor sparse_mm(torch::Tensor sparse, torch::Tensor dense) {
    std::cout << "Sparse tensor shape: " << sparse.sizes() << std::endl;
    std::cout << "Dense tensor shape: " << dense.sizes() << std::endl;
    return sparse;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mm", &sparse_mm, "Sparse matrix multiplication");
}