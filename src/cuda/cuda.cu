#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
using namespace std;

// __global__ void device_hello_world() {
//     printf("Hello world from x.%d y.%d z.%d!\n", threadIdx.x, threadIdx.y, threadIdx.z);
// }
// dim3 block(2, 2, 2);    // x*y*z <= 1024
// dim3 grid(1, 1, 1);
// device_hello_world<<<grid, block>>>();
// cudaDeviceSynchronize();

#define BLOCK_SIZE 16


//TODO: Optimize, for now it's processing the whole matrix everytime
__global__ void generate_identity_matrix(double* matrix, int size){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int limit = size * 2;
    if (row < size && col < limit) {
        if (row == col - size) matrix[row * limit + col] = 1;
    }
}

__global__ void scale_row(double* matrix, int size, int selected_row){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int limit = size * 2;
    double scale = matrix[selected_row * limit + selected_row];
    if (row == selected_row && col < limit) {
        matrix[row * limit + col] /= scale;
    }
}

__global__ void clear_column(double* matrix, int size, int selected_col){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int limit = size * 2;
    double scale = matrix[row * limit + selected_col];
    if (row != selected_col && col < limit && row < size) {
        matrix[row * limit + col] -= scale * matrix[selected_col * limit + col];
    }
}

void print_matrix(double* mat, int x, int y, int x_offset){
    cout << y << endl;
    for(int i=0; i < y; ++i)
    {
        int offset = i * x;
        for(int j = x_offset; j < x; ++j)
        {
            cout << mat[offset + j] << " ";
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
    double* mat = new double[n * n_double];
    double* cuda_mat = new double[n * n_double];
    cudaMalloc(&cuda_mat, n * n_double * sizeof(double));
    cudaDeviceSynchronize();
    
    int offset;
    for (int row = 0; row < n; ++row) {
        offset = row * n_double;
        for (int col = 0; col < n; col++){
            cin >> mat[offset + col];
        }
    }
    auto start = chrono::high_resolution_clock::now();
    cudaMemcpy(cuda_mat, mat, n * n_double * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(n / block.x * 2, n / block.y);

    generate_identity_matrix<<<grid, block>>>(cuda_mat, n);

    for (int row = 0; row < n; ++row){
        scale_row<<<grid, block>>>(cuda_mat, n, row);
        clear_column<<<grid, block>>>(cuda_mat, n, row);
    }

    cudaMemcpy(mat, cuda_mat, n * n_double * sizeof(double), cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_taken = end - start;
    cout << time_taken.count() << " Seconds" << std::endl;

    print_matrix(mat, n_double, n, n);

    delete[] mat;
    cudaFree(cuda_mat);

    return 0;
}