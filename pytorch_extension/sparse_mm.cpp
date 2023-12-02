#include <torch/extension.h>
#include <iostream>
#include <ATen/native/Resize.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/ATen.h>
#include <ATen/native/CompositeRandomAccessorCommon.h>
#include "utils.h"
#include "multiplication_utils.h"

template <typename scalar_t>
torch::Tensor sparse_matmul_kernel(
    torch::Tensor output,
    const torch::Tensor mat1,
    const torch::Tensor &mat2)
{
    /*
      Computes  the sparse-sparse matrix multiplication between `mat1` and `mat2`, which are sparse tensors in COO format.
    */

    auto M = mat1.size(0);
    auto N = mat2.size(1);

    const auto mat1_csr = mat1.to_sparse_csr();
    const auto mat2_csr = mat2.to_sparse_csr();

    auto mat1_crow_indices_ptr = at::native::StridedRandomAccessor<int64_t>(
        mat1_csr.crow_indices().data_ptr<int64_t>(),
        mat1_csr.crow_indices().stride(-1));
    auto mat1_col_indices_ptr = at::native::StridedRandomAccessor<int64_t>(
        mat1_csr.col_indices().data_ptr<int64_t>(),
        mat1_csr.col_indices().stride(-1));
    auto mat1_values_ptr = at::native::StridedRandomAccessor<scalar_t>(
        mat1_csr.values().data_ptr<scalar_t>(),
        mat1_csr.values().stride(-1));
    auto mat2_crow_indices_ptr = at::native::StridedRandomAccessor<int64_t>(
        mat2_csr.crow_indices().data_ptr<int64_t>(),
        mat2_csr.crow_indices().stride(-1));
    auto mat2_col_indices_ptr = at::native::StridedRandomAccessor<int64_t>(
        mat2_csr.col_indices().data_ptr<int64_t>(),
        mat2_csr.col_indices().stride(-1));
    auto mat2_values_ptr = at::native::StridedRandomAccessor<scalar_t>(
        mat2_csr.values().data_ptr<scalar_t>(),
        mat2_csr.values().stride(-1));

    const auto nnz = _csr_matmult_maxnnz(
        M,
        N,
        mat1_crow_indices_ptr,
        mat1_col_indices_ptr,
        mat2_crow_indices_ptr,
        mat2_col_indices_ptr);

    auto output_indices = output._indices();
    auto output_values = output._values();

    torch::Tensor output_indptr = at::empty({M + 1}, at::kLong);
    at::native::resize_output(output_indices, {2, nnz});
    at::native::resize_output(output_values, nnz);

    torch::Tensor output_row_indices = output_indices.select(0, 0);
    torch::Tensor output_col_indices = output_indices.select(0, 1);

    // TODO: replace with a CSR @ CSC kernel for better performance.
    _csr_matmult(
        M,
        N,
        mat1_crow_indices_ptr,
        mat1_col_indices_ptr,
        mat1_values_ptr,
        mat2_crow_indices_ptr,
        mat2_col_indices_ptr,
        mat2_values_ptr,
        output_indptr.data_ptr<int64_t>(),
        output_col_indices.data_ptr<int64_t>(),
        output_values.data_ptr<scalar_t>());

    csr_to_coo(M, output_indptr.data_ptr<int64_t>(), output_row_indices.data_ptr<int64_t>());

    output._coalesced_(true);

    return output;
}

// TODO: solve the following error: output matrix will be all zero
torch::Tensor sparse_sparse_matmul_cpu(const torch::Tensor &mat1_, const torch::Tensor &mat2_)
{ 
    std::cout << "sparse_sparse_matmul_cpu" << std::endl;
    TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
    TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
    TORCH_CHECK(mat1_.dim() == 2);
    TORCH_CHECK(mat2_.dim() == 2);
    TORCH_CHECK(mat1_.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", mat1_.dense_dim(), "D values");
    TORCH_CHECK(mat2_.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", mat2_.dense_dim(), "D values");

    TORCH_CHECK(
        mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
        mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

    TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
                "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

    auto output = at::native::empty_like(mat1_);
    output.sparse_resize_and_clear_({mat1_.size(0), mat2_.size(1)}, mat1_.sparse_dim(), 0);

    auto answer = AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(mat1_.scalar_type(), "sparse_matmul", [&]() -> at::Tensor {
      return sparse_matmul_kernel<scalar_t>(output, mat1_.coalesce(), mat2_.coalesce());
    });

    return answer;
}

torch::Tensor sparse_mm(torch::Tensor sparse_matrix_0, torch::Tensor sparse_matrix_1)
{
    TORCH_CHECK(sparse_matrix_0.is_sparse(), "sparse_matrix_0 must be a sparse tensor");
    TORCH_CHECK(sparse_matrix_1.is_sparse(), "sparse_matrix_1 must be a sparse tensor");
    
    torch::Tensor answer = sparse_sparse_matmul_cpu(sparse_matrix_0, sparse_matrix_1);
    return answer;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_mm", &sparse_mm, "Sparse matrix multiplication");
}