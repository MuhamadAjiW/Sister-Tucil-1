# ----------------ARGUMENTS----------------
# Change test case as needed from the function call
TEST_CASE ?= 2048.txt


# ----------------VARIABLES----------------
# General Variables
OUTPUT_FOLDER = bin
TEST_FOLDER = test_cases
RESULT_FOLDER = results
GCC_OPTIMIZATION_FLAGS = -O3 -march=native -fallow-store-data-races -fno-semantic-interposition

# Serial Variables
SERIAL_EXECUTABLE = serial.exe
SERIAL_SRC = serial.cpp

# MPI Variables
MPI_EXECUTABLE = mpi.exe
MPI_SRC = mpi.cpp
MPI_BIN = $(MSMPI_BIN)
MPI_INC = $(MSMPI_INC)
MPI_LIB32 = $(MSMPI_LIB32)
MPI_LIB64 = $(MSMPI_LIB64)
MPI_FLAGS = $(GCC_OPTIMIZATION_FLAGS) -I "${MPI_INC}" -L "${MPI_LIB64}" -lmsmpi

# OpenMP Variables
OPENMP_EXECUTABLE = openmp.exe
OPENMP_SRC = openmp.cpp
OPENMP_FLAGS = $(GCC_OPTIMIZATION_FLAGS) -fopenmp

# CUDA Variables
CUDA_EXECUTABLE = cuda.exe
CUDA_SRC = cuda.cu



# ----------------SCRIPTS----------------
# Compile Scripts
all: build exec_all
build: build_serial build_parallel
	@echo "Everything is compiled' to execute the program"

build_serial:
	@echo "Compiling serial program..."
	@g++ src/serial/$(SERIAL_SRC) -o $(OUTPUT_FOLDER)/$(SERIAL_EXECUTABLE)

build_parallel: build_mpi build_openmp build_cuda
build_mpi:
	@echo "Compiling MPI program..."
	@g++ src/mpi/$(MPI_SRC) ${MPI_FLAGS} -o $(OUTPUT_FOLDER)/$(MPI_EXECUTABLE)
build_openmp:
	@echo "Compiling OpenMP program..."
	@g++ src/open-mp/$(OPENMP_SRC) $(OPENMP_FLAGS) -o $(OUTPUT_FOLDER)/$(OPENMP_EXECUTABLE)
build_cuda:
	@echo "Compiling CUDA program..."
	@nvcc src/cuda/$(CUDA_SRC) -o $(OUTPUT_FOLDER)/$(CUDA_EXECUTABLE)

# Execute programs
exec_all: exec_serial exec_mpi exec_openmp exec_openmp
exec_serial:
	@echo "Executing Serial program..."
	@cd $(OUTPUT_FOLDER) && ./$(SERIAL_EXECUTABLE) < ../$(TEST_FOLDER)/$(TEST_CASE) > ../$(RESULT_FOLDER)/serial_$(TEST_CASE)
exec_mpi:
	@echo "Executing MPI program..."
	@cd $(OUTPUT_FOLDER) && mpiexec $(MPI_EXECUTABLE)  < ../$(TEST_FOLDER)/$(TEST_CASE) > ../$(RESULT_FOLDER)/mpi_$(TEST_CASE)
exec_openmp:
	@echo "Executing OpenMP program..."
	@cd $(OUTPUT_FOLDER) && ./$(OPENMP_EXECUTABLE) < ../$(TEST_FOLDER)/$(TEST_CASE) > ../$(RESULT_FOLDER)/openmp_$(TEST_CASE)
exec_cuda:
	@echo "Executing CUDA program..."
	@cd $(OUTPUT_FOLDER) && ./$(CUDA_EXECUTABLE) < ../$(TEST_FOLDER)/$(TEST_CASE) > ../$(RESULT_FOLDER)/cuda_$(TEST_CASE)

# Combination for testing
serial	: build_serial exec_serial
mpi		: build_mpi exec_mpi
openmp	: build_openmp exec_openmp
cuda	: build_cuda exec_cuda

# Clean
clean:
	@rm -rf bin results && mkdir bin results