#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <stdio.h>

const int THREADS = 32;

template <typename index_t, typename scalar_t>
void thread_matmul(const int start_row, const int end_row, const index_t* Ap, const index_t* Aj, const scalar_t* Ax, const index_t* Bp, const index_t* Bj, const scalar_t* Bx, std::vector<std::vector<index_t>> &next, std::vector<std::vector<scalar_t>> &sums, std::vector<index_t> &head, std::vector<int> &length) {
    std::thread::id this_id = std::this_thread::get_id();
    std::cout << "thread " << this_id << std::endl;
    for (int i = start_row; i < end_row; i++) {
        index_t jj_start = Ap[i];
        index_t jj_end = Ap[i + 1];
        for (index_t jj = jj_start; jj < jj_end; jj++)
        {
            index_t j = Aj[jj];
            scalar_t v = Ax[jj];

            index_t kk_start = Bp[j];
            index_t kk_end = Bp[j + 1];
            for (index_t kk = kk_start; kk < kk_end; kk++)
            {
                index_t k = Bj[kk];
                sums[i][k] += v * Bx[kk];

                if (next[i][k] == -1)
                {
                    next[i][k] = head[i];
                    head[i] = k;
                    length[i]++;
                }
            }
        }
    }
}

template <typename index_t, typename scalar_t>
void _csr_matmult(int n_row, int n_col, index_t Ap[], index_t Aj[], scalar_t Ax[], index_t Bp[], index_t Bj[], scalar_t Bx[], index_t Cp[], index_t Cj[], scalar_t Cx[])
{
    // define thread objects
    std::thread threads[THREADS];

    // 計算每個thread要處理的row數量
    int each_row_thread = std::ceil(static_cast<float>(n_row) / THREADS);
    // 但是因為每個thread要處理的row數量不一定相同，所以要計算每個thread真的要處理的row數量 (可能是因為不整除或是row小於thread數量)
    std::vector<int> actual_row_thread(THREADS, 0);
    std::vector<int> start_row(THREADS, 0);
    std::vector<int> end_row(THREADS, 0);
    int remain_row = n_row;
    for (int i = 0; i < THREADS; i++)
    {
        // Calculate actual_row_thread
        actual_row_thread[i] = (remain_row >= each_row_thread) ? each_row_thread : remain_row;
        remain_row -= actual_row_thread[i];

        // Calculate start_row
        start_row[i] = (i == 0) ? 0 : start_row[i - 1] + actual_row_thread[i - 1];

        // Calculate end_row
        end_row[i] = (i == 0) ? actual_row_thread[i] : end_row[i - 1] + actual_row_thread[i];
    }

    std::vector<std::vector<index_t>> next(n_row, std::vector<index_t>(n_col, -1));
    std::vector<std::vector<scalar_t>> sums(n_row, std::vector<scalar_t>(n_col, 0));
    std::vector<index_t> head(n_row, -2);
    std::vector<int> length(n_row, 0);

    int nnz = 0;
    for (int t = 0; t < THREADS; t++) {
        // use thread_matmul to calculate next, sums, head, length
        // thread_matmul<index_t, scalar_t>(start_row[t], end_row[t], Ap, Aj, Ax, Bp, Bj, Bx, next, sums, head, length);
        threads[t] = std::thread(thread_matmul<index_t, scalar_t>, start_row[t], end_row[t], Ap, Aj, Ax, Bp, Bj, Bx, std::ref(next), std::ref(sums), std::ref(head), std::ref(length));
    }
    for (int t = 0; t < THREADS; t++) {
        threads[t].join();
    }

    for (int t = 0; t < THREADS; t++) {
        for (int i = start_row[t]; i < end_row[t]; i++)
        {
            for (int jj = 0; jj < length[i]; jj++)
            {

                Cj[nnz] = head[i];
                Cx[nnz] = sums[i][head[i]];
                nnz++;

                index_t temp = head[i];
                head[i] = next[i][head[i]];

                next[i][temp] = -1; // clear arrays
                sums[i][temp] = 0;
            }

            Cp[i + 1] = nnz;
        }
    }

    printf("Cp: ");
    for (int i = 0; i < 4; i++)
    {
        printf("%ld ", Cp[i]);
    }
    printf("\n");
    printf("Cj: ");
    for (int i = 0; i < 4; i++)
    {
        printf("%ld ", Cj[i]);
    }
    printf("\n");
    printf("Cx: ");
    for (int i = 0; i < 4; i++)
    {
        printf("%ld ", Cx[i]);
    }
    printf("\n");
}
int main()
{
    int64_t row_ptr_A[4] = {0, 1, 3, 3};
    int64_t collindices_A[4] = {2, 0, 2};
    int64_t data_A[4] = {3, 4, 5};
    int64_t row_ptr_B[4] = {0, 1, 1, 3};
    int64_t collindices_B[4] = {1, 0, 1};
    int64_t data_B[4] = {4, 3, 5};
    int n_row = 3;
    int n_col = 3;
    int64_t cp[4] = {0, 0, 0, 0};
    int64_t cj[4] = {0, 0, 0, 0};
    int64_t cx[4] = {0, 0, 0, 0};

    _csr_matmult<int64_t, int64_t>(n_row, n_col, row_ptr_A, collindices_A, data_A, row_ptr_B, collindices_B, data_B, cp, cj, cx);
}