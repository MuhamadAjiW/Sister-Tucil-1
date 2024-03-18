// gcc mp.c --openmp -o mp

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
using namespace std;

// Macros so the overhead is in the compiler
#define THREAD_COUNT 16

void print_matrix(double** mat, int x, int y, int x_offset){
    cout << y << endl;
    for(int i=0; i < y; ++i)
    {
        for(int j = x_offset; j < x; ++j)
        {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
}

int main(void) {
    double** mat = NULL;

    // Read matrix information in main thread
    int n;

    cin >> n;
    int n_double = 2 * n;
    int n_rows = n / THREAD_COUNT;
    int local_size = n_double * n_rows;

    // Why even initialize it with 2n x 2n in the serial code? That's not very memory efficient
    mat = new double*[n];
    for (int row = 0; row < n; ++row) {
        mat[row] = new double[n_double]();
        for (int col = 0; col < n; col++){
            cin >> mat[row][col];
        }            
    }
    auto start = chrono::high_resolution_clock::now();


    double scale = 0;
    #pragma omp parallel num_threads(THREAD_COUNT)
    {
        int tid = omp_get_thread_num();
        int start_row = n_rows * tid;
        int end_row = start_row + n_rows;


        // Identity matrix
        for (int row = start_row; row < end_row; ++row){
            mat[row][row + n] = 1;
        }

        for (int row = 0; row < n; row++){
            #pragma omp barrier
            #pragma omp single
            {
                scale = mat[row][row];
            }

            for (int col  = start_row * 2; col < end_row * 2; col++){
                mat[row][col] /= scale;
            }
            
            #pragma omp barrier

            for (int row2 = start_row; row2 < end_row; row2++){
                if(row2 == row) continue;

                double multiplier = mat[row2][row];
                for (int col = 0; col < n_double; col++){
                    mat[row2][col] -= mat[row][col] * multiplier;
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_taken = end - start;
    cout << time_taken.count() << " Seconds" << std::endl;

    print_matrix(mat, n_double, n, n);

    for (int i = 0; i < n; ++i)
    {
        delete[] mat[i];
    }
    delete[] mat;

    return 0;
}