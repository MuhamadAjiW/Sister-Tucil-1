OUTPUT_FOLDER = bin

# ----------------VARIABLES----------------
# Assign environment as needed
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
MPI_FLAGS = -g -fdiagnostics-color=always -I ${MPI_INC} -L ${MPI_LIB64} -lmsmpi

# OpenMP Variables
OPENMP_EXECUTABLE = openmp.exe
OPENMP_SRC = openmp.cpp
OPENMP_FLAGS = -g -fdiagnostics-color=always -fopenmp

# CUDA Variables
CUDA_EXECUTABLE = cuda.exe
CUDA_SRC = cuda.cu



# ----------------SCRIPTS----------------
# Compile Scripts
all: serial parallel
	@echo "Everything is compiled, use 'make exec_<serial/mpi/openmp/cuda>' to execute the program"
parallel: mpi openmp cuda

serial:
	@echo "Compiling serial program..."
	@g++ src/serial/$(SERIAL_SRC) -o $(OUTPUT_FOLDER)/$(SERIAL_EXECUTABLE)
mpi:
	@echo "Compiling MPI program..."
	@g++ src/open-mpi/$(MPI_SRC) ${MPI_FLAGS} -o $(OUTPUT_FOLDER)/$(MPI_EXECUTABLE)
openmp:
	@echo "Compiling OpenMP program..."
	@g++ src/open-mp/$(OPENMP_SRC) $(OPENMP_FLAGS) -o $(OUTPUT_FOLDER)/$(OPENMP_EXECUTABLE)
cuda:
	@echo "Compiling CUDA program..."
	@nvcc src/cuda/$(CUDA_SRC) -o $(OUTPUT_FOLDER)/$(CUDA_EXECUTABLE)

# Execute programs
exec_serial:
	@cd $(OUTPUT_FOLDER) && ./$(SERIAL_EXECUTABLE)
exec_mpi:
	@cd $(OUTPUT_FOLDER) && mpiexec $(MPI_EXECUTABLE)
exec_openmp:
	@cd $(OUTPUT_FOLDER) && ./$(OPENMP_EXECUTABLE)
exec_cuda:
	@cd $(OUTPUT_FOLDER) && ./$(CUDA_EXECUTABLE)

# Clean
clean:
	@rm -rf bin && mkdir bin