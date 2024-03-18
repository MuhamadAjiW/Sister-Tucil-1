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
    int i = 0, j = 0, k = 0, n = 0;

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


    #pragma omp parallel for num_threads(THREAD_COUNT)
    for(i = 0; i < n; ++i){
        mat[i][i + n] = 1;
    }

    // Unparallelizable
    for(i = n - 1; i > 0; --i){
        if(mat[i-1][1] < mat[i][1]){
            double* temp = mat[i];
            mat[i] = mat[i-1];
            mat[i-1] = temp;
        }
    }

    // Reducing To Diagonal Matrix
    for(i = 0; i < n; ++i){
        #pragma parallel for num_threads(THREAD_COUNT)
        for(j = 0; j < n; ++j){
            if(j != i){
                double multiplier = mat[j][i] / mat[i][i];
                for(k = 0; k < n*2; ++k){
                    mat[j][k] -= mat[i][k] * multiplier;
                }
            }
        }
    }
    
    // Reducing To Unit Matrix
    #pragma omp parallel for num_threads(THREAD_COUNT)
    for(i = 0; i < n; ++i){
        double multiplier = mat[i][i];
        for(j = 0; j < 2*n; ++j){
            mat[i][j] = mat[i][j] / multiplier;
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