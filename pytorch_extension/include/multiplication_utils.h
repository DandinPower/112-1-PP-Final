#pragma once
#include <omp.h>
#include <torch/extension.h>
#include <utils.h>
#include <logger.h>

void csr_to_coo(const int64_t n_row, const int64_t Ap[], int64_t Bi[]) {
    logger.startTest("csr_to_coo");
    for (const auto i : c10::irange(n_row)) {
        for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++) {
            Bi[jj] = i;
        }
    }
    logger.endTest("csr_to_coo");
}

template <typename index_t_ptr = int64_t *>
int64_t _csr_matmult_maxnnz(const int64_t n_row, const int64_t n_col,
                            const index_t_ptr Ap, const index_t_ptr Aj,
                            const index_t_ptr Bp, const index_t_ptr Bj) {
    logger.startTest("csr_matmult_maxnnz");
    int64_t nnz = 0;

    // for (const auto i : c10::irange(n_row))
    std::vector<int64_t> mask(n_col, -1);
    for (int i = 0; i < n_row; i++) {
        int64_t row_nnz = 0;
        for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++) {
            int64_t j = Aj[jj];
            for (int64_t kk = Bp[j]; kk < Bp[j + 1]; kk++) {
                int64_t k = Bj[kk];
                if (mask[k] != i) {
                    mask[k] = i;
                    row_nnz++;
                }
            }
        }
        nnz += row_nnz;
    }
    logger.endTest("csr_matmult_maxnnz");
    return nnz;
}

template <typename index_t_ptr = int64_t *>
int64_t _csr_matmult_maxnnz_parallel(const int64_t n_row, const int64_t n_col,
                                     const index_t_ptr Ap, const index_t_ptr Aj,
                                     const index_t_ptr Bp,
                                     const index_t_ptr Bj) {
    logger.startTest("csr_matmult_maxnnz");
    int64_t nnz = 0;

#pragma omp parallel for reduction(+ : nnz)
    // for (const auto i : c10::irange(n_row))
    for (int i = 0; i < n_row; i++) {
        int64_t row_nnz = 0;
        std::vector<int64_t> mask(n_col, -1);
        for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++) {
            int64_t j = Aj[jj];
            for (int64_t kk = Bp[j]; kk < Bp[j + 1]; kk++) {
                int64_t k = Bj[kk];
                if (mask[k] != i) {
                    mask[k] = i;
                    row_nnz++;
                }
            }
        }
        nnz += row_nnz;
    }
    logger.endTest("csr_matmult_maxnnz");
    return nnz;
}