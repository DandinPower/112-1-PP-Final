#pragma once
#include <torch/extension.h>
#include <iostream>
#include <omp.h>
#include <ATen/native/Resize.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/ATen.h>
#include <ATen/native/CompositeRandomAccessorCommon.h>
#include "omp_utils.h"
#include "omp_multiplication_utils.h"

template <typename index_t_ptr, typename scalar_t_ptr>
void omp_csr_matmult(
    const int64_t n_row,
    const int64_t n_col,
    const index_t_ptr Ap,
    const index_t_ptr Aj,
    const scalar_t_ptr Ax,
    const index_t_ptr Bp,
    const index_t_ptr Bj,
    const scalar_t_ptr Bx,
    typename index_t_ptr::value_type Cp[],
    typename index_t_ptr::value_type Cj[],
    typename scalar_t_ptr::value_type Cx[])
{

  using index_t = typename index_t_ptr::value_type;
  using scalar_t = typename scalar_t_ptr::value_type;

  std::vector<int64_t> nnzs(n_row, 0);

  Cp[0] = 0;

  #pragma omp parallel 
  {
    std::vector<index_t>  next(n_col, -1);
    std::vector<scalar_t> sums(n_col, 0);

    #pragma omp for ordered schedule(static, 1)
    for (const int64_t i = 0; i < n_row; i++)
    {
      index_t head   = -2;
      index_t length = 0;
      index_t jj_start = Ap[i];
      index_t jj_end   = Ap[i + 1];
      for (const index_t jj = jj_start; jj < jj_end; jj++)
      {
        index_t  j = Aj[jj];
        scalar_t v = Ax[jj];
        index_t  kk_start = Bp[j];
        index_t  kk_end   = Bp[j + 1];
        for (const index_t kk = kk_start; kk < kk_end; kk++)
        {
          index_t k = Bj[kk];
          sums[k] += v * Bx[kk];
          if (next[k] == -1)
          {
            next[k] = head;
            head = k;
            length++;
          }
        }
      }

      nnzs[i] = length;

      #pragma omp barrier
      #pragma omp ordered
      if (i > 0)
        nnzs[i] += nnzs[i - 1];

      index_t nnz = nnzs[i];
      for (const index_t jj = nnz - length; jj < nnz; jj++)
      {
        Cj[jj] = head;
        Cx[jj] = sums[head];

        index_t temp = head;
        head = next[head];

        next[temp] = -1;
        sums[temp] = 0;
      }

      auto col_indices_accessor = at::native::StridedRandomAccessor<int64_t>(Cj + nnz - length, 1);
      auto val_accessor = at::native::StridedRandomAccessor<scalar_t>(Cx + nnz - length, 1);
      auto kv_accessor = at::native::CompositeRandomAccessorCPU<
          decltype(col_indices_accessor), decltype(val_accessor)>(col_indices_accessor, val_accessor);

      std::sort(kv_accessor, kv_accessor + length, [](const auto &lhs, const auto &rhs) -> bool
                { return get_first(lhs) < get_first(rhs); });

      Cp[i + 1] = nnz;
    }
  }
}

template <typename scalar_t>
torch::Tensor omp_sparse_matmul_kernel(
    torch::Tensor output,
    const torch::Tensor &mat1,
    const torch::Tensor &mat2)
{
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

    const auto nnz = omp_csr_matmult_maxnnz(
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

#if DEBUG == 1
    std::cout << "Output: "<< std::endl;
    omp_view<int64_t>(nnz*2, output._indices().data_ptr<int64_t>());
    omp_view<scalar_t>(nnz, output._values().data_ptr<scalar_t>());
#endif

    auto indices = torch::empty({2, nnz}, torch::kLong);
    auto values = torch::empty({nnz}, torch::kFloat);
    omp_copy_value<int64_t>(nnz*2, indices.data_ptr<int64_t>(), output._indices().data_ptr<int64_t>());
    omp_copy_value<scalar_t>(nnz, values.data_ptr<scalar_t>(), output._values().data_ptr<scalar_t>());
    return torch::sparse_coo_tensor(indices, values, {M, N});
}

torch::Tensor sparse_sparse_matmul_omp(const torch::Tensor &mat1_, const torch::Tensor &mat2_)
{ 
    TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
    TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
    TORCH_CHECK(mat1_.dim() == 2);
    TORCH_CHECK(mat2_.dim() == 2);
    TORCH_CHECK(mat1_.dense_dim() == 0, "sparse_sparse_matmul_omp: scalar values expected, got ", mat1_.dense_dim(), "D values");
    TORCH_CHECK(mat2_.dense_dim() == 0, "sparse_sparse_matmul_omp: scalar values expected, got ", mat2_.dense_dim(), "D values");

    TORCH_CHECK(
        mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
        mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

    TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
                "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

    auto output = at::native::empty_like(mat1_);
    output.sparse_resize_and_clear_({mat1_.size(0), mat2_.size(1)}, mat1_.sparse_dim(), 0);

    auto answer = AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(mat1_.scalar_type(), "sparse_matmul", [&]() -> at::Tensor {
      return omp_sparse_matmul_kernel<scalar_t>(output, mat1_.coalesce(), mat2_.coalesce());
    });

    return answer;
}