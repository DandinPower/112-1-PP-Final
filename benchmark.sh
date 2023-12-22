# PATH to save the benchmark results
# VERBOSE is the flag to print the benchmark results, if VERBOSE=1, it will print the results to PROFILING_RESULT_PATH
# BENCHMARK_RESULT_PATH is the path to save the benchmark results
# PROFILING_RESULT_PATH is the path to save the profiling results in the extension
VERBOSE=1
BENCHMARK_RESULT_PATH=logs/python_benchmark_results.log
PROFILING_RESULT_PATH=logs/extension_profiling_results.log

# Matrix Size parameters
# MATRIX_START_SIZE means the start size of the matrix
# MATRIX_END_SIZE means the end size of the matrix
# MATRIX_STEP_SIZE means the step size of the matrix, each step will increase the matrix size by MATRIX_STEP_SIZE
MATRIX_START_SIZE=1
MATRIX_END_SIZE=4096
MATRIX_STEP_SIZE=2 

# Density parameters
# DENSITY_START_SIZE means the start size of the density, the actual density will divide by 10
# DENSITY_END_SIZE means the end size of the density, the actual density will divide by 10
# DENSITY_STEP_SIZE means the step size of the density, each step will increase the density by multiply DENSITY_STEP_SIZE
DENSITY_START_SIZE=1
DENSITY_END_SIZE=8
DENSITY_STEP_SIZE=2

# Number of threads to use for parallelization
# Careful Setting this number should check your core all have the same performance
# For example, intel performance core and efficiency core will have different performance
# NUM_THREADS_START means the start number of threads
# NUM_THREADS_END means the end number of threads
# NUM_THREADS_STEP means the step number of threads, each step will increase the number of threads by NUM_THREADS_STEP
NUM_THREADS_START=1
NUM_THREADS_END=16
NUM_THREADS_STEP=2

# NUM_ITERATIONS means the number of iterations to run for each benchmark
# It will run NUM_ITERATIONS times for each benchmark and calculate the average time
NUM_ITERATIONS=10

# Run the benchmark
python benchmark.py > $PROFILING_RESULT_PATH \
    --verbose $VERBOSE \
    --size_start $MATRIX_START_SIZE \
    --size_end $MATRIX_END_SIZE \
    --size_step $MATRIX_STEP_SIZE \
    --density_start $DENSITY_START_SIZE \
    --density_end $DENSITY_END_SIZE \
    --density_step $DENSITY_STEP_SIZE \
    --num_threads_start $NUM_THREADS_START \
    --num_threads_end $NUM_THREADS_END \
    --num_threads_step $NUM_THREADS_STEP \
    --num_iterations $NUM_ITERATIONS \
    --log_file $BENCHMARK_RESULT_PATH