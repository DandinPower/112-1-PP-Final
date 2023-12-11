#include <iostream>
#include <vector>
#include <stdio.h>
void _csr_matmult(int n_row, int n_col, int Ap[], int Aj[], int Ax[], int Bp[], int Bj[], int Bx[], int Cp[], int Cj[], int Cx[])
{
    std::vector<std::vector<int>> next(n_row, std::vector<int>(n_col, -1));
    std::vector<std::vector<int>> sums(n_row, std::vector<int>(n_col, 0));
    int nnz = 0;
    std::vector<int> head(n_row, -2);
    int length[n_col] = {0};
    for (int i = 0; i < n_row; i++)
    {
        int jj_start = Ap[i];
        int jj_end = Ap[i + 1];
        for (int jj = jj_start; jj < jj_end; jj++)
        {
            int j = Aj[jj];
            int v = Ax[jj];

            int kk_start = Bp[j];
            int kk_end = Bp[j + 1];
            for (int kk = kk_start; kk < kk_end; kk++)
            {
                int k = Bj[kk];

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
    for (int i = 0; i < n_row; i++)
    {
        for (int jj = 0; jj < length[i]; jj++)
        {

            Cj[nnz] = head[i];
            Cx[nnz] = sums[i][head[i]];
            nnz++;

            int temp = head[i];
            head[i] = next[i][head[i]];

            next[i][temp] = -1; // clear arrays
            sums[i][temp] = 0;
        }
        Cp[i + 1] = nnz;
    }
    
    for (int i = 0; i < 4; i++)
    {
        printf("Cp:%d \n", Cp[i]);
    }
    for (int i = 0; i < 4; i++)
    {
        printf("Cj:%d \n", Cj[i]);
    }
    for (int i = 0; i < 4; i++)
    {
        printf("Cx:%d \n", Cx[i]);
    }
}
int main()
{
    int row_ptr_A[4] = {0, 1, 3, 3};
    int collindices_A[4] = {2, 0, 2};
    int data_A[4] = {3, 4, 5};

    int row_ptr_B[4] = {0, 1, 1, 3};
    int collindices_B[4] = {1, 0, 1};
    int data_B[4] = {4, 3, 5};
    int n_row = 3;
    int n_col = 3;
    int cp[4] = {0, 0, 0, 0};
    int cj[4] = {0, 0, 0, 0};
    int cx[4] = {0, 0, 0, 0};
    _csr_matmult(n_row, n_col, row_ptr_A, collindices_A, data_A, row_ptr_B, collindices_B, data_B, cp, cj, cx);
}