#pragma once
#include <ATen/ATen.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessorCommon.h>
#include <ATen/native/Resize.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <config.h>
#include <multiplication_utils.h>
#include <torch/extension.h>
#include <utils.h>

#include <iostream>
#include <thread>

template <typename index_t, typename scalar_t, typename index_t_ptr,
          typename scalar_t_ptr>
void thread_matmul(const int start_row, const int end_row, const index_t_ptr Ap,
                   const index_t_ptr Aj, const scalar_t_ptr Ax,
                   const index_t_ptr Bp, const index_t_ptr Bj,
                   const scalar_t_ptr Bx,
                   std::vector<std::vector<index_t>> &next,
                   std::vector<std::vector<scalar_t>> &sums,
                   std::vector<index_t> &head, std::vector<index_t> &length) {
    for (int i = start_row; i < end_row; i++) {
        index_t jj_start = Ap[i];
        index_t jj_end = Ap[i + 1];
        for (index_t jj = jj_start; jj < jj_end; jj++) {
            index_t j = Aj[jj];
            scalar_t v = Ax[jj];

            index_t kk_start = Bp[j];
            index_t kk_end = Bp[j + 1];
            for (index_t kk = kk_start; kk < kk_end; kk++) {
                index_t k = Bj[kk];
                sums[i][k] += v * Bx[kk];

                if (next[i][k] == -1) {
                    next[i][k] = head[i];
                    head[i] = k;
                    length[i]++;
                }
            }
        }
    }
}

template <typename index_t, typename scalar_t, typename index_t_ptr,
          typename scalar_t_ptr>
void thread_put_answer(const int start_row, const int end_row,
                       const index_t* Cp, index_t* Cj, scalar_t* Cx,
                       std::vector<std::vector<index_t>> &next,
                       std::vector<std::vector<scalar_t>> &sums,
                       std::vector<index_t> &head,
                       std::vector<index_t> &length) {
    for (int i = start_row; i < end_row; i++) {
        int previous_nnz = Cp[i];
        for (int jj = 0; jj < length[i]; jj++) {
            Cj[previous_nnz] = head[i];
            Cx[previous_nnz] = sums[i][head[i]];
            previous_nnz++;

            index_t temp = head[i];
            head[i] = next[i][head[i]];

            next[i][temp] = -1;
            sums[i][temp] = 0;
        }

        auto col_indices_accessor = at::native::StridedRandomAccessor<int64_t>(
            Cj + previous_nnz - length[i], 1);
        auto val_accessor = at::native::StridedRandomAccessor<scalar_t>(
            Cx + previous_nnz - length[i], 1);
        auto kv_accessor = at::native::CompositeRandomAccessorCPU<
            decltype(col_indices_accessor), decltype(val_accessor)>(
            col_indices_accessor, val_accessor);

        std::sort(kv_accessor, kv_accessor + length[i],
                  [](const auto &lhs, const auto &rhs) -> bool {
                      return get_first(lhs) < get_first(rhs);
                  });
    }
}

template <typename index_t_ptr, typename scalar_t_ptr>
void _csr_matmult_std_thread(const int num_threads, const int64_t n_row,
                             const int64_t n_col, const index_t_ptr Ap,
                             const index_t_ptr Aj, const scalar_t_ptr Ax,
                             const index_t_ptr Bp, const index_t_ptr Bj,
                             const scalar_t_ptr Bx,
                             typename index_t_ptr::value_type Cp[],
                             typename index_t_ptr::value_type Cj[],
                             typename scalar_t_ptr::value_type Cx[]) {
    using index_t = typename index_t_ptr::value_type;
    using scalar_t = typename scalar_t_ptr::value_type;
    // define thread objects
    // std::thread threads[THREADS];

    std::vector<std::thread> threads(num_threads);

    // 計算每個thread要處理的row數量
    int each_row_thread = std::ceil(static_cast<float>(n_row) / num_threads);
    // 但是因為每個thread要處理的row數量不一定相同，所以要計算每個thread真的要處理的row數量
    // (可能是因為不整除或是row小於thread數量)
    std::vector<int> actual_row_thread(num_threads, 0);
    std::vector<int> start_row(num_threads, 0);
    std::vector<int> end_row(num_threads, 0);
    int remain_row = n_row;
    int thread_count = 0;
    for (int i = 0; i < num_threads; i++) {
        thread_count++;
        actual_row_thread[i] =
            (remain_row >= each_row_thread) ? each_row_thread : remain_row;
        remain_row -= actual_row_thread[i];
        start_row[i] =
            (i == 0) ? 0 : start_row[i - 1] + actual_row_thread[i - 1];
        end_row[i] = (i == 0) ? actual_row_thread[i]
                              : end_row[i - 1] + actual_row_thread[i];
        if (remain_row == 0) {
            break;
        }
    }

    std::vector<std::vector<index_t>> next(n_row,
                                           std::vector<index_t>(n_col, -1));
    std::vector<std::vector<scalar_t>> sums(n_row,
                                            std::vector<scalar_t>(n_col, 0));
    std::vector<index_t> head(n_row, -2);
    std::vector<index_t> length(n_row, 0);
    Cp[0] = 0;

    for (int t = 0; t < thread_count; t++) {
        threads[t] = std::thread(
            thread_matmul<index_t, scalar_t, index_t_ptr, scalar_t_ptr>,
            start_row[t], end_row[t], Ap, Aj, Ax, Bp, Bj, Bx, std::ref(next),
            std::ref(sums), std::ref(head), std::ref(length));
    }
    for (int t = 0; t < thread_count; t++) {
        threads[t].join();
    }

    int64_t tempNnz = 0;
    for (int i = 0; i < n_row; i++) {
        tempNnz += length[i];
        Cp[i + 1] = tempNnz;
    }

    for (int t = 0; t < thread_count; t++) {
        threads[t] = std::thread(
            thread_put_answer<index_t, scalar_t, index_t_ptr, scalar_t_ptr>,
            start_row[t], end_row[t], Cp, Cj, Cx, std::ref(next),
            std::ref(sums), std::ref(head), std::ref(length));
    }
    for (int t = 0; t < thread_count; t++) {
        threads[t].join();
    }
}

template <typename scalar_t>
torch::Tensor sparse_matmul_kernel_std_thread(const torch::Tensor &mat1,
                                              const torch::Tensor &mat2,
                                              const int num_threads) {
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

    _csr_matmult_std_thread(
        num_threads, M, N, mat1_crow_indices_ptr, mat1_col_indices_ptr,
        mat1_values_ptr, mat2_crow_indices_ptr, mat2_col_indices_ptr,
        mat2_values_ptr, output_indptr.data_ptr<int64_t>(),
        output_col_indices.data_ptr<int64_t>(), values.data_ptr<scalar_t>());

    csr_to_coo(M, output_indptr.data_ptr<int64_t>(),
               output_row_indices.data_ptr<int64_t>());
    return torch::sparse_coo_tensor(indices, values, {M, N});
}

torch::Tensor sparse_sparse_matmul_cpu_std_thread(const torch::Tensor &mat1_,
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
            return sparse_matmul_kernel_std_thread<scalar_t>(
                mat1_.coalesce(), mat2_.coalesce(), num_threads);
        });

    return answer;
}