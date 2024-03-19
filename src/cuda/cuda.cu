#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

// __global__ void device_hello_world() {
//     printf("Hello world from x.%d y.%d z.%d!\n", threadIdx.x, threadIdx.y, threadIdx.z);
// }
// dim3 block(2, 2, 2);    // x*y*z <= 1024
// dim3 grid(1, 1, 1);
// device_hello_world<<<grid, block>>>();
// cudaDeviceSynchronize();

#define BLOCK_SIZE 32


//TODO: Optimize, for now it's processing the whole matrix everytime
__global__ void generate_identity_matrix(double* matrix, int row_limit, int col_limit){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if(row == 0 && col == 0) printf("Generating identity matrix\n");
    
    if (row < row_limit && col < col_limit) {
        if (row == col - row_limit) matrix[row * col_limit + col] = 1;
    }
}

__global__ void scale_row(double* matrix, int row_limit, int col_limit, int selected_row){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if(row == 0 && col == 0) printf("Scalling row %d\n", selected_row);
    
    double scale = matrix[selected_row * col_limit + selected_row];
    if (row == selected_row && col < col_limit && row != col) {
        matrix[row * col_limit + col] /= scale;
    }
}

__global__ void scale_pivot(double* matrix, int row_limit, int col_limit, int selected_row){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if(row == 0 && col == 0) printf("Scalling row %d\n", selected_row);
    
    if (row == selected_row && row == col) {
        matrix[row * col_limit + col] = 1;
    }
}

__global__ void reduce_rows(double* matrix, int row_limit, int col_limit, int selected_col){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if(row == 0 && col == 0) printf("Clearing column %d\n", selected_col);
    
    double scale = matrix[row * col_limit + selected_col];
    if (row != selected_col && col < col_limit && row < row_limit && col != selected_col) {
        matrix[row * col_limit + col] -= scale * matrix[selected_col * col_limit + col];
    }
}

__global__ void clear_column(double* matrix, int row_limit, int col_limit, int selected_col){  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if(row == 0 && col == 0) printf("Clearing column %d\n", selected_col);
    
    double scale = matrix[row * col_limit + selected_col];
    if (row != selected_col && col < col_limit && row < row_limit && col == selected_col) {
        matrix[row * col_limit + col] = 0;
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
    cudaEventRecord(start, 0);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(n / block.x * 2, n / block.y);

    cudaMemcpy(cuda_mat, mat, matrix_size, cudaMemcpyHostToDevice);
    generate_identity_matrix<<<grid, block>>>(cuda_mat, n, n_double);

    for (int row = 0; row < n; ++row){
        scale_row<<<grid, block>>>(cuda_mat, n, n_double, row);
        scale_pivot<<<grid, block>>>(cuda_mat, n, n_double, row);
        reduce_rows<<<grid, block>>>(cuda_mat, n, n_double, row);
        clear_column<<<grid, block>>>(cuda_mat, n, n_double, row);
    }

    cudaMemcpy(mat, cuda_mat, matrix_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cout << time / 1000 << " Seconds" << std::endl;
    
    print_matrix(mat, n_double, n, n);

    delete[] mat;
    cudaFree(cuda_mat);

    return 0;
}