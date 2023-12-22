from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from src.extension import ExtensionHandler
from src.models import DnnClassifier, SparseDnnClassifier
from src.prune import prune_model
from src.matmul_strategy import MatmulStrategy
from src.utils import create_config_list_by_mul_step

DEVICE = torch.device('cpu')

def prune_model_by_trained_model(density: float, model_path: str, pruned_model_path: str):
    model = DnnClassifier()
    model.load_state_dict(torch.load(model_path))
    prune_model(model, density)
    torch.save(model.state_dict(), pruned_model_path)

def train(train_loader: DataLoader, test_loader: DataLoader, epoch: int, model_save_path: str):
    print('Start training...')
    model = DnnClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch):
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Accuracy: {accuracy:.2f}%')
    torch.save(model.state_dict(), model_save_path)

def _inference(test_loader: DataLoader, model: SparseDnnClassifier, strategy, thread_num: int):
    model.matmul_strategy.set_matmul_strategy(strategy)
    model.matmul_strategy.set_thread_num(thread_num)
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            model.logger.reset()
            outputs = model.forward(images)
            total += model.logger.get_total()
    return total

def inference_benchmark(test_loader: DataLoader, args: argparse.Namespace):
    print('Start mnist inference benchmark...')
    model = SparseDnnClassifier()
    model.matmul_strategy = MatmulStrategy()

    strategy_set = [
        ('builtin', ExtensionHandler.sparse_mm),
        ('parallel_structure', ExtensionHandler.parallel_structure_sparse_mm),
        ('openmp', ExtensionHandler.openmp_sparse_mm),
        ('openmp_mem_effi', ExtensionHandler.openmp_mem_effi_sparse_mm),
        ('std_thread', ExtensionHandler.std_thread_sparse_mm)
    ]
    
    density_set = create_config_list_by_mul_step(args.density_start / 10, (args.density_end + 1) / 10, args.density_step)
    threads_set = create_config_list_by_mul_step(args.num_threads_start, args.num_threads_end + 1, args.num_threads_step)
    
    with open(args.log_file, 'w') as f:
        for density in density_set:
            prune_model_by_trained_model(density, args.model_save_path, args.pruned_model_save_path)
            model.load_state_dict(args.pruned_model_save_path, 6)
            for thread_num in threads_set:
                print(f'density=[ {density} ], num_threads= [ {thread_num} ]')
                output_string = f'density=[ {density} ], num_threads= [ {thread_num} ]\n'
                output_string += f'strategy, duration(ms)\n'
                output_string += "-" * 50 + "\n"
                for strategy_name, strategy in strategy_set:
                    t = 0
                    for _ in range(args.num_iterations):
                        if strategy_name == 'builtin' or strategy_name == 'parallel_structure':
                            t += _inference(test_loader, model, strategy, -1)
                        else:
                            t += _inference(test_loader, model, strategy, thread_num)
                    t /= args.num_iterations
                    output_string += f'{strategy_name},{t*1000:.3f}\n'
                output_string += "-" * 50 + "\n"
                f.write(output_string)

def main():
    parser = argparse.ArgumentParser(description='MNIST Benchmark')
    parser.add_argument('--verbose', type=int, help='whether to print the extension results')
    parser.add_argument('--re_train', type=int, help='whether to retrain the model')
    parser.add_argument('--mnist_data_save_path', type=str, help='the path to save the mnist data')
    parser.add_argument('--model_save_path', type=str, help='the path to save the model')
    parser.add_argument('--pruned_model_save_path', type=str, help='the path to save the pruned model')
    parser.add_argument('--epoch', type=int, help='the number of epochs')
    parser.add_argument('--density_start', type=int, help='the starting density of the sparse matrix')
    parser.add_argument('--density_end', type=int, help='the ending density of the sparse matrix')
    parser.add_argument('--density_step', type=int, help='the step density of the sparse matrix')
    parser.add_argument('--num_threads_start', type=int, help='the starting number of threads')
    parser.add_argument('--num_threads_end', type=int, help='the ending number of threads')
    parser.add_argument('--num_threads_step', type=int, help='the step number of threads')
    parser.add_argument('--num_iterations', type=int, help='the number of runs for each configuration')
    parser.add_argument('--log_file', type=str, help='the log file to store the benchmark results')
    args = parser.parse_args()
    args.re_train = args.re_train == 1
    train_dataset = MNIST(root=args.mnist_data_save_path, train=True, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = MNIST(root=args.mnist_data_save_path, train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=8)

    if args.re_train:
        train(train_loader, test_loader, args.epoch, args.model_save_path)
    else:
        try: 
            torch.load(args.model_save_path) is not None, 'Please train the model first.'
        except:
            print('Please train the model first.')
            exit(1)

    ExtensionHandler.set_verbose(args.verbose)

    inference_benchmark(test_loader, args)

if __name__ == '__main__':
    main()