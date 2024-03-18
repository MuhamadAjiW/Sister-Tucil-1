// gcc mp.c --openmp -o mp

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
using namespace std;

// Macros so the overhead is in the compiler
#define THREAD_COUNT 8

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
    // Read matrix information in main thread
    int n;

    cin >> n;
    int n_double = 2 * n;

    // Why even initialize it with 2n x 2n in the serial code? That's not very memory efficient
    double** mat = new double*[n];
    for (int row = 0; row < n; ++row) {
        mat[row] = new double[n_double]();
        for (int col = 0; col < n; col++){
            cin >> mat[row][col];
        }
    }
    auto start = chrono::high_resolution_clock::now();

    // Identity matrix
    #pragma omp simd
    for (int row = 0; row < n; ++row){
        mat[row][row + n] = 1;
    }

    for (int row = 0; row < n; ++row){
        double scale = mat[row][row];

        // Single row
        #pragma omp simd
        for (int col  = 0; col < n_double; col++){
            mat[row][col] /= scale;
        }
        
        // Propagate line to other rows 
        #pragma omp parallel for num_threads(THREAD_COUNT)
        for (int row2 = 0; row2 < n; row2++){
            if(row2 == row) continue;

            double multiplier = mat[row2][row];

            #pragma omp simd
            for (int col = 0; col < n_double; col++){
                mat[row2][col] -= mat[row][col] * multiplier;
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