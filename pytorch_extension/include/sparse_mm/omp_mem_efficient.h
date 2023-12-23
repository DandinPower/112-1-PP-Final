#pragma once
#include <ATen/ATen.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessorCommon.h>
#include <ATen/native/Resize.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <logger.h>
#include <multiplication_utils.h>
#include <torch/extension.h>
#include <utils.h>

#include <iostream>

namespace omp_mem_efficient {

template <typename index_t_ptr, typename scalar_t_ptr>
void _csr_matmult(const int num_threads, const int64_t n_row, const int64_t n_col, const index_t_ptr Ap,
                 const index_t_ptr Aj, const scalar_t_ptr Ax,
                 const index_t_ptr Bp, const index_t_ptr Bj,
                 const scalar_t_ptr Bx, typename index_t_ptr::value_type Cp[],
                 typename index_t_ptr::value_type Cj[],
                 typename scalar_t_ptr::value_type Cx[]) {
    using index_t = typename index_t_ptr::value_type;
    using scalar_t = typename scalar_t_ptr::value_type;

    omp_set_num_threads(num_threads);
    std::vector<int64_t> nnzs(n_row, 0);

    Cp[0] = 0;

#pragma omp parallel
    {
        std::vector<index_t> next(n_col, -1);
        std::vector<scalar_t> sums(n_col, 0);

#pragma omp for ordered schedule(static, 1)
        for (int64_t i = 0; i < n_row; i++) {
            index_t head = -2;
            index_t length = 0;
            index_t jj_start = Ap[i];
            index_t jj_end = Ap[i + 1];
            for (index_t jj = jj_start; jj < jj_end; jj++) {
                index_t j = Aj[jj];
                scalar_t v = Ax[jj];
                index_t kk_start = Bp[j];
                index_t kk_end = Bp[j + 1];
                for (index_t kk = kk_start; kk < kk_end; kk++) {
                    index_t k = Bj[kk];
                    sums[k] += v * Bx[kk];
                    if (next[k] == -1) {
                        next[k] = head;
                        head = k;
                        length++;
                    }
                }
            }

            nnzs[i] = length;

#pragma omp ordered
            if (i > 0) nnzs[i] += nnzs[i - 1];

            index_t nnz = nnzs[i];
            for (index_t jj = nnz - length; jj < nnz; jj++) {
                Cj[jj] = head;
                Cx[jj] = sums[head];

                index_t temp = head;
                head = next[head];

                next[temp] = -1;
                sums[temp] = 0;
            }

            auto col_indices_accessor =
                at::native::StridedRandomAccessor<int64_t>(Cj + nnz - length,
                                                           1);
            auto val_accessor = at::native::StridedRandomAccessor<scalar_t>(
                Cx + nnz - length, 1);
            auto kv_accessor = at::native::CompositeRandomAccessorCPU<
                decltype(col_indices_accessor), decltype(val_accessor)>(
                col_indices_accessor, val_accessor);

            std::sort(kv_accessor, kv_accessor + length,
                      [](const auto &lhs, const auto &rhs) -> bool {
                          return get_first(lhs) < get_first(rhs);
                      });

            Cp[i + 1] = nnz;
        }
    }
}

template <typename scalar_t>
torch::Tensor sparse_matmul_kernel(const torch::Tensor &mat1,
                                   const torch::Tensor &mat2,
                                   const int num_threads) {
    logger.startTest("convert_to_csr");
    auto M = mat1.size(0);
    auto N = mat2.size(1);

    const auto mat1_csr = mat1.to_sparse_csr();
    const auto mat2_csr = mat2.to_sparse_csr();
    logger.endTest("convert_to_csr");

    auto mat1_crow_indices_ptr = at::native::StridedRandomAccessor<int64_t>(
        mat1_csr.crow_indices().data_ptr<int64_t>(),
        mat1_csr.crow_indices().stride(-1));
    auto mat1_col_indices_ptr = at::native::StridedRandomAccessor<int64_t>(
        mat1_csr.col_indices().data_ptr<int64_t>(),
        mat1_csr.col_indices().stride(-1));
    auto mat1_values_ptr = at::native::StridedRandomAccessor<scalar_t>(
        mat1_csr.values().data_ptr<scalar_t>(), mat1_csr.values().stride(-1));
    auto mat2_crow_indices_ptr = at::native::StridedRandomAccessor<int64_t>(
        mat2_csr.crow_indices().data_ptr<int64_t>(),
        mat2_csr.crow_indices().stride(-1));
    auto mat2_col_indices_ptr = at::native::StridedRandomAccessor<int64_t>(
        mat2_csr.col_indices().data_ptr<int64_t>(),
        mat2_csr.col_indices().stride(-1));
    auto mat2_values_ptr = at::native::StridedRandomAccessor<scalar_t>(
        mat2_csr.values().data_ptr<scalar_t>(), mat2_csr.values().stride(-1));

    const auto nnz = _csr_matmult_maxnnz_parallel(
        M, N, mat1_crow_indices_ptr, mat1_col_indices_ptr,
        mat2_crow_indices_ptr, mat2_col_indices_ptr);

    torch::Tensor output_indptr = at::empty({M + 1}, at::kLong);

    torch::Tensor indices = torch::empty({2, nnz}, torch::kLong);
    torch::Tensor values = torch::empty({nnz}, torch::kFloat);

    torch::Tensor output_row_indices = indices.select(0, 0);
    torch::Tensor output_col_indices = indices.select(0, 1);

    _csr_matmult(num_threads, M, N, mat1_crow_indices_ptr, mat1_col_indices_ptr,
                 mat1_values_ptr, mat2_crow_indices_ptr, mat2_col_indices_ptr,
                 mat2_values_ptr, output_indptr.data_ptr<int64_t>(),
                 output_col_indices.data_ptr<int64_t>(),
                 values.data_ptr<scalar_t>());

    csr_to_coo(M, output_indptr.data_ptr<int64_t>(),
               output_row_indices.data_ptr<int64_t>());

    logger.startTest("create_sparse_tensor");
    auto answer = torch::sparse_coo_tensor(indices, values, {M, N});
    logger.endTest("create_sparse_tensor");
    return answer;
}

torch::Tensor sparse_sparse_matmul_cpu(const torch::Tensor &mat1_,
                                       const torch::Tensor &mat2_,
                                       const int num_threads) {
    TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
    TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
    TORCH_CHECK(mat1_.dim() == 2);
    TORCH_CHECK(mat2_.dim() == 2);
    TORCH_CHECK(mat1_.dense_dim() == 0,
                "sparse_sparse_matmul_cpu: scalar values expected, got ",
                mat1_.dense_dim(), "D values");
    TORCH_CHECK(mat2_.dense_dim() == 0,
                "sparse_sparse_matmul_cpu: scalar values expected, got ",
                mat2_.dense_dim(), "D values");
    TORCH_CHECK(mat1_.size(1) == mat2_.size(0),
                "mat1 and mat2 shapes cannot be multiplied (", mat1_.size(0),
                "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1),
                ")");

    TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(), "mat1 dtype ",
                mat1_.scalar_type(), " does not match mat2 dtype ",
                mat2_.scalar_type());

    auto answer = AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        mat1_.scalar_type(), "sparse_matmul", [&]() -> at::Tensor {
            return sparse_matmul_kernel<scalar_t>(
                mat1_.coalesce(), mat2_.coalesce(), num_threads);
        });

    return answer;
}
}  // namespace omp_mem_efficient