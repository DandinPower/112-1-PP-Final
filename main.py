import multiprocessing

# Number of cores in the system
num_cores = multiprocessing.cpu_count()

def benchmark_example():
    """
    An example of how to benchmark a single test configuration.
    """
    from src.benchmark import show_benchmark_results, benchmark_by_config_list, generate_benchmark_configurations
    config_list = generate_benchmark_configurations(size_start=10, size_end=510, size_step= 200, density_start=1, density_end=10, density_step=4, num_threads_start=1, num_threads_end=num_cores, num_threads_step=6)
    results = benchmark_by_config_list(config_list, num_runs=2)
    show_benchmark_results(results)

def main():
    benchmark_example()

if __name__ == '__main__':
    main()