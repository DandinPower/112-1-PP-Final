import multiprocessing

# Number of cores in the system
num_cores = multiprocessing.cpu_count()

def benchmark_example():
    """
    An example of how to benchmark multiple test configuration.
    """
    from src.benchmark import show_benchmark_results, log_benchmark_results, benchmark_by_config_list, generate_benchmark_configurations
    config_list = generate_benchmark_configurations(size_start=8, size_end=4096+1, size_step= 2, density_start=1, density_end=10, density_step=2, num_threads_start=1, num_threads_end=num_cores+1, num_threads_step=2)
    results = benchmark_by_config_list(config_list, num_runs=2)
    # show_benchmark_results(results)
    log_benchmark_results(results, "logs/python_benchmark_results.log")

def benchmark_example_single():
    """
    An example of how to benchmark a single test configuration.
    """
    from src.benchmark import show_benchmark_results, log_benchmark_results, benchmark_by_config_list, generate_benchmark_configurations
    config_list = generate_benchmark_configurations(size_start=450, size_end=500, size_step= 100, density_start=4, density_end=5, density_step=4, num_threads_start=16, num_threads_end=32, num_threads_step=30)
    results = benchmark_by_config_list(config_list, num_runs=1)
    # show_benchmark_results(results)
    log_benchmark_results(results, "logs/python_benchmark_results.log")

def main():
    benchmark_example()
    # benchmark_example_single()

if __name__ == '__main__':
    main()