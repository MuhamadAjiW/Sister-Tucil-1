#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 32
#define MAX_BLOCK_SIZE 1024

__global__ void generate_identity_matrix(double* matrix, int row_limit, int col_limit){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < row_limit) {
        matrix[row * col_limit + row + row_limit] = 1;
    }
}

__global__ void scale_row(double* matrix, int col_limit, int selected_row){  
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    double scale = matrix[selected_row * col_limit + selected_row];
    if (col < col_limit && selected_row != col) {
        matrix[selected_row * col_limit + col] /= scale;
    }
}

__global__ void scale_pivot(double* matrix, int col_limit, int selected_row){  
    matrix[selected_row * col_limit + selected_row] = 1;
}

__global__ void reduce_rows(double* matrix, int row_limit, int col_limit, int selected_col){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    double scale = matrix[row * col_limit + selected_col];
    if (row != selected_col && col < col_limit && row < row_limit && col != selected_col) {
        matrix[row * col_limit + col] -= scale * matrix[selected_col * col_limit + col];
    }
}

__global__ void clear_column(double* matrix, int row_limit, int col_limit, int selected_col){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < row_limit && row != selected_col) {
        matrix[row * col_limit + selected_col] = 0;
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
    int matrix_size = n * n_double * sizeof(double);

    // Don't collect error because it's an overhead lol, just trust on these codes
    // cudaError_t error;

    // Why even initialize it with 2n x 2n in the serial code? That's not very memory efficient
    double* mat = new double[n * n_double];
    double* cuda_mat = new double[n * n_double];
    cudaMalloc(&cuda_mat, matrix_size);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int offset;
    for (int row = 0; row < n; ++row) {
        offset = row * n_double;
        for (int col = 0; col < n; col++){
            cin >> mat[offset + col];
        }
    }
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil(n / block.x * 2) , ceil(n / block.y));

    dim3 block_inline_col(MAX_BLOCK_SIZE, 1);
    dim3 grid_inline_col(ceil(n / block.x * 2), 1);

    dim3 block_inline_row(1, MAX_BLOCK_SIZE);
    dim3 grid_inline_row(1, ceil(n / block.y));

    cudaMemcpy(cuda_mat, mat, matrix_size, cudaMemcpyHostToDevice);

    // Timer starts at identity matrix initialization  
    cudaEventRecord(start, 0);

    generate_identity_matrix<<<grid_inline_row, block_inline_row>>>(cuda_mat, n, n_double);

    for (int row = 0; row < n; ++row){
        // Even though they're the same operations
        // scale row and scale pivot has to be separated because of race conditions
        // same deal with clear column and reduce rows

        scale_row<<<grid_inline_col, block_inline_col>>>(cuda_mat, n_double, row);
        scale_pivot<<<1, 1>>>(cuda_mat, n_double, row);
        reduce_rows<<<grid, block>>>(cuda_mat, n, n_double, row);
        clear_column<<<grid_inline_row, block_inline_row>>>(cuda_mat, n, n_double, row);
    }

    cudaMemcpy(mat, cuda_mat, matrix_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << time / 1000 << " Seconds" << std::endl;
    
    print_matrix(mat, n_double, n, n);

    delete[] mat;
    cudaFree(cuda_mat);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}