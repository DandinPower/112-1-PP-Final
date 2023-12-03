from src.benchmark import generate_and_benchmark_configurations, show_benchmark_results

def main():
    results = generate_and_benchmark_configurations(size_start=5, size_end=500, size_step=50, density_start=1, density_end=10, density_step=4)
    show_benchmark_results(results)

if __name__ == '__main__':
    main()