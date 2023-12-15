#pragma once
#include <torch/extension.h>
#include <omp.h>
#include <omp_utils.h>

void omp_csr_to_coo(const int64_t n_row, const int64_t Ap[], int64_t Bi[])
{
  for (const auto i : c10::irange(n_row))
  {
    for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++)
    {
      Bi[jj] = i;
    }
  }
}

template <typename index_t_ptr = int64_t *>
int64_t omp_csr_matmult_maxnnz(
    const int64_t n_row,
    const int64_t n_col,
    const index_t_ptr Ap,
    const index_t_ptr Aj,
    const index_t_ptr Bp,
    const index_t_ptr Bj)
{
  std::vector<int64_t> mask(n_col, -1);
  int64_t nnz = 0;

  for (int64_t i = 0; i < n_row; i++)
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