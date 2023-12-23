from src.extension import ExtensionHandler
import argparse

def benchmark(args):
    from src.benchmark import log_benchmark_results, benchmark_by_config_list, generate_benchmark_configurations
    config_list = generate_benchmark_configurations(size_start=args.size_start, size_end=args.size_end+1, size_step=args.size_step, density_start=args.density_start, density_end=args.density_end+1, density_step=args.density_step, num_threads_start=args.num_threads_start, num_threads_end=args.num_threads_end+1, num_threads_step=args.num_threads_step)
    results = benchmark_by_config_list(config_list, num_runs=args.num_iterations)
    log_benchmark_results(results, args.log_file)

def main():
    parser = argparse.ArgumentParser(description='SPMM Benchmark')
    parser.add_argument('--verbose', type=int, help='whether to print the extension results')
    parser.add_argument('--size_start', type=int, help='the starting size of the sparse matrix')
    parser.add_argument('--size_end', type=int, help='the ending size of the sparse matrix')
    parser.add_argument('--size_step', type=int, help='the step size of the sparse matrix')
    parser.add_argument('--density_start', type=int, help='the starting density of the sparse matrix')
    parser.add_argument('--density_end', type=int, help='the ending density of the sparse matrix')
    parser.add_argument('--density_step', type=int, help='the step density of the sparse matrix')
    parser.add_argument('--num_threads_start', type=int, help='the starting number of threads')
    parser.add_argument('--num_threads_end', type=int, help='the ending number of threads')
    parser.add_argument('--num_threads_step', type=int, help='the step number of threads')
    parser.add_argument('--num_iterations', type=int, help='the number of runs for each configuration')
    parser.add_argument('--log_file', type=str, help='the log file to store the benchmark results')
    args = parser.parse_args()
    ExtensionHandler.set_verbose(args.verbose)
    benchmark(args)

if __name__ == '__main__':
    main()