#pragma once
#include <torch/extension.h>
#include "utils.h"

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
void csr_to_coo(const int64_t n_row, const int64_t Ap[], int64_t Bi[])
{
    /*
      Expands a compressed row pointer into a row indices array
      Inputs:
        `n_row` is the number of rows in `Ap`
        `Ap` is the row pointer

      Output:
        `Bi` is the row indices
    */
    for (const auto i : c10::irange(n_row))
    {
        for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++)
        {
            Bi[jj] = i;
        }
    }
}

template <typename index_t_ptr = int64_t *>
int64_t _csr_matmult_maxnnz(
    const int64_t n_row,
    const int64_t n_col,
    const index_t_ptr Ap,
    const index_t_ptr Aj,
    const index_t_ptr Bp,
    const index_t_ptr Bj)
{
    /*
      Compute needed buffer size for matrix `C` in `C = A@B` operation.

      The matrices should be in proper CSR structure, and their dimensions
      should be compatible.
    */
    std::vector<int64_t> mask(n_col, -1);
    int64_t nnz = 0;
    for (const auto i : c10::irange(n_row))
    {
        int64_t row_nnz = 0;

        for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++)
        {
            int64_t j = Aj[jj];
            for (int64_t kk = Bp[j]; kk < Bp[j + 1]; kk++)
            {
                int64_t k = Bj[kk];
                if (mask[k] != i)
                {
                    mask[k] = i;
                    row_nnz++;
                }
            }
        }
        int64_t next_nnz = nnz + row_nnz;
        nnz = next_nnz;
    }
    return nnz;
}

template<typename index_t_ptr, typename scalar_t_ptr>
void _csr_matmult(
    const int64_t n_row,
    const int64_t n_col,
    const index_t_ptr Ap,
    const index_t_ptr Aj,
    const scalar_t_ptr Ax,
    const index_t_ptr Bp,
    const index_t_ptr Bj,
    const scalar_t_ptr Bx,
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename index_t_ptr::value_type Cp[],
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename index_t_ptr::value_type Cj[],
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename scalar_t_ptr::value_type Cx[]) {
  /*
    Compute CSR entries for matrix C = A@B.

    The matrices `A` and 'B' should be in proper CSR structure, and their dimensions
    should be compatible.

    Inputs:
      `n_row`         - number of row in A
      `n_col`         - number of columns in B
      `Ap[n_row+1]`   - row pointer
      `Aj[nnz(A)]`    - column indices
      `Ax[nnz(A)]     - nonzeros
      `Bp[?]`         - row pointer
      `Bj[nnz(B)]`    - column indices
      `Bx[nnz(B)]`    - nonzeros
    Outputs:
      `Cp[n_row+1]` - row pointer
      `Cj[nnz(C)]`  - column indices
      `Cx[nnz(C)]`  - nonzeros

    Note:
      Output arrays Cp, Cj, and Cx must be preallocated
  */
  using index_t = typename index_t_ptr::value_type;
  using scalar_t = typename scalar_t_ptr::value_type;

  std::vector<index_t> next(n_col, -1);
  std::vector<scalar_t> sums(n_col, 0);

  int64_t nnz = 0;

  Cp[0] = 0;

  for (const auto i : c10::irange(n_row)) {
    index_t head = -2;
    index_t length = 0;

    index_t jj_start = Ap[i];
    index_t jj_end = Ap[i + 1];
    for (const auto jj : c10::irange(jj_start, jj_end)) {
      index_t j = Aj[jj];
      scalar_t v = Ax[jj];

      index_t kk_start = Bp[j];
      index_t kk_end = Bp[j + 1];
      for (const auto kk : c10::irange(kk_start, kk_end)) {
        index_t k = Bj[kk];

        sums[k] += v * Bx[kk];

        if (next[k] == -1) {
          next[k] = head;
          head = k;
          length++;
        }
      }
    }

    for (C10_UNUSED const auto jj : c10::irange(length)) {

      // NOTE: the linked list that encodes col indices
      // is not guaranteed to be sorted.
      Cj[nnz] = head;
      Cx[nnz] = sums[head];
      nnz++;

      index_t temp = head;
      head = next[head];

      next[temp] = -1; // clear arrays
      sums[temp] = 0;
    }

    // Make sure that col indices are sorted.
    // TODO: a better approach is to implement a CSR @ CSC kernel.
    // NOTE: Cx arrays are expected to be contiguous!
    auto col_indices_accessor = at::native::StridedRandomAccessor<int64_t>(Cj + nnz - length, 1);
    auto val_accessor = at::native::StridedRandomAccessor<scalar_t>(Cx + nnz - length, 1);
    auto kv_accessor = at::native::CompositeRandomAccessorCPU<
      decltype(col_indices_accessor), decltype(val_accessor)
    >(col_indices_accessor, val_accessor);

    std::sort(kv_accessor, kv_accessor + length, [](const auto& lhs, const auto& rhs) -> bool {
        return get_first(lhs) < get_first(rhs);
    });

    Cp[i + 1] = nnz;
  }
}