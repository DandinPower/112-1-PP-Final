# MNIST Benchmark Parameters

# MNIST_DATA_SAVE_PATH is the path to save the mnist data
# MODEL_SAVE_PATH is the path to save the model
# PRUNED_MODEL_SAVE_PATH is the path to save the pruned model
MNIST_DATA_SAVE_PATH=mnist/data
MODEL_SAVE_PATH=mnist/model.pth
PRUNED_MODEL_SAVE_PATH=mnist/pruned_model.pth

# VERBOSE is the flag to print the benchmark results, if VERBOSE=1, it will print the results to terminal
# RE_TRAIN is the flag to retrain the model, if you set RE_TRAIN=0, please make sure the model is already trained
# EPOCH is the number of epochs
# DENSITY_START means the starting density of the sparse matrix
# DENSITY_END means the ending density of the sparse matrix
# DENSITY_STEP means the step density of the sparse matrix, each step will increase the density by multiply DENSITY_STEP
# NUM_THREADS_START means the starting number of threads
# NUM_THREADS_END means the ending number of threads
# NUM_THREADS_STEP means the step number of threads, each step will increase the number of threads by multiply NUM_THREADS_STEP
VERBOSE=0
RE_TRAIN=1
EPOCH=10
DENSITY_START=1
DENSITY_END=8
DENSITY_STEP=2
NUM_THREADS_START=1
NUM_THREADS_END=16
NUM_THREADS_STEP=2

# NUM_ITERATIONS means the number of runs for each configuration
# LOG_FILE is the log file to store the benchmark results
NUM_ITERATIONS=10
LOG_FILE=logs/mnist_benchmark_results.log

# Run the mnist benchmark
python mnist_benchmark.py \
    --verbose $VERBOSE \
    --re_train $RE_TRAIN \
    --mnist_data_save_path $MNIST_DATA_SAVE_PATH \
    --model_save_path $MODEL_SAVE_PATH \
    --pruned_model_save_path $PRUNED_MODEL_SAVE_PATH \
    --epoch $EPOCH \
    --density_start $DENSITY_START \
    --density_end $DENSITY_END \
    --density_step $DENSITY_STEP \
    --num_threads_start $NUM_THREADS_START \
    --num_threads_end $NUM_THREADS_END \
    --num_threads_step $NUM_THREADS_STEP \
    --num_iterations $NUM_ITERATIONS \
    --log_file $LOG_FILE