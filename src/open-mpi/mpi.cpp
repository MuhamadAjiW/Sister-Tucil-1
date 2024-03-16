// mpicc mpi.c -o mpi

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

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

// Main functionality not split into functions to reduce function calls
int main(void) {
    // MPI vars
    // Variables are matched with register length, hence long long
    int world_size = 0;
    int world_rank = 0;

    //init MPI with queue for non blocking communications
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // For some reason this does not work using cpp's new, hence malloc
    MPI_Request* request_queue = (MPI_Request*) malloc (sizeof(MPI_Request) * world_size);
    // MPI_Request* request_queue = (MPI_Request*) malloc (sizeof(MPI_Request) * world_size);
    
    // Matrix vars, changed to row and col to improve readability
    int i = 0, row = 0, col = 0, n = 0;
    double* mat = NULL;

    // Initialize the queue with null
    for (i = 0; i < world_size; i++){
        request_queue[i] = MPI_REQUEST_NULL;
    }

    // Read matrix information in world_rank 0
    if (world_rank == 0){
        cin >> n;
        int n_double = 2 * n;

        // Why even initialize it with 2n x 2n in the serial code? That's not very memory efficient
        mat = new double[n_double * n];
        for (row = 0; row < n; ++row) {
            int offset = row * n_double;

            for (col = 0; col < n; col++){
                cin >> mat[offset + col];
            }            
        }

        for (row = 0; row < n; row++){
            mat[row * n_double + row + n] = 1;
        }
        
    }

    // Initialize local matrix
    // Lots of variables are here to avoid recalculation; storing over recalculating, memory is cheap
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int n_double = n * 2;
    int n_rows = n / world_size;
    int local_size = n_double * n_rows;
    double* local_mat = new double[local_size];
    double* pivot_row = new double[n_double];
    int start_row = world_rank * n_rows;
    int end_row = start_row + n_rows;

    // Scatter rows
    MPI_Scatter(
        mat,
        local_size,
        MPI_DOUBLE,

        local_mat,
        local_size,
        MPI_DOUBLE,

        0, MPI_COMM_WORLD
    );

    // Actual operation for each processes
    // Max is end_row for each world rank since that's all the row's needed for
    for (row = 0; row < end_row; row++){
        
        // Spread using batch
        // e.g for 32 with 4 processes it would be 0, 1, 2, 3
        // Round robin wouldn't do any better in this situation since indices does not hold value
        if(row >= start_row && row < end_row){
            int local_row = row - start_row;

            // Pivot calculation to the right
            int offset = local_row * n_double;
            double pivot = local_mat[offset + row];
            for (col = row; col < n_double; col++){
                local_mat[offset + col] /= pivot;
            }

            // Send resulting row to processes with rows below, nonblocking since we are not waiting for anything in return
            for (i = 0; i < world_size; i++){
                if(world_rank == i) continue;

                MPI_Isend(
                    // Data to be sent
                    local_mat + offset,
                    n_double,
                    MPI_DOUBLE,

                    // Receiver Info
                    i,

                    // Communication lines
                    0, MPI_COMM_WORLD, &request_queue[i]
                );
            }

            // Update column values using the given pivot
            for (i = 0; i < n_rows; i++){
                if(i == local_row) continue;

                int elimination_offset = i * n_double;
                double scale = local_mat[elimination_offset + row];

                // Update column values using the given pivot
                for (col = 0; col < n_double; col++){
                    local_mat[elimination_offset + col] -= local_mat[offset + col] * scale;
                }
            }
        }

        // Receive pivot rows if not in batch
        else{
            // Blocking receive
            MPI_Recv(
                // Data to be received
                pivot_row,
                n_double,
                MPI_DOUBLE,

                // Sender Info
                row / n_rows,
                
                // Communication lines
                0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            for (i = 0; i < n_rows; i++){
                int offset = i * n_double;
                double scale = local_mat[offset + row];

                // Update column values using the given pivot
                for (col = 0; col < n_double; col++){
                    local_mat[offset + col] -= pivot_row[col] * scale;
                }
            }
        }
    }

    
    for (row = end_row; row < n; row++){
        // Blocking receive
        MPI_Recv(
            // Data to be received
            pivot_row,
            n_double,
            MPI_DOUBLE,

            // Sender Info
            row / n_rows,
            
            // Communication lines
            0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        for (i = 0; i < n_rows; i++){
            int offset = i * n_double;
            double scale = local_mat[offset + row];

            // Update column values using the given pivot
            for (col = 0; col < n_double; col++){
                local_mat[offset + col] -= pivot_row[col] * scale;
            }
        }
    }
    // if(world_rank == 0) print_matrix(local_mat, n_double, n_rows, 0);

    
    // Gather time
    MPI_Gather(
        // Data to be gathered
        local_mat,
        local_size,
        MPI_DOUBLE,

        // Gather info
        mat,
        local_size,
        MPI_DOUBLE,

        // Communication lines
        0, MPI_COMM_WORLD
    );

    // Check for results
    if(world_rank == 0) print_matrix(mat, n_double, n, n);

    // Clean up
    if(world_rank == 0) delete[] mat;
    delete[] local_mat;
    free(request_queue);

    MPI_Finalize();

    return 0;
}