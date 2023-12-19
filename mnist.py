from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from src.extension import ExtensionHandler
from src.utils import SparseMatrixTestConfiguration, generate_sparse_matrix
import time
import matplotlib.pyplot as plt

EPOCH = 1
DATA_PATH = 'mnist/data'
FULL_MODEL_PATH = 'mnist/weight/model.pth'
PRUNED_MODEL_PATH = 'mnist/weight/pruned_model.pth'

DEVICE = torch.device('cpu')

train_dataset = MNIST(root=DATA_PATH, train=True, transform=ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = MNIST(root=DATA_PATH, train=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=8)

class DnnClassifier(nn.Module):
    def __init__(self):
        super(DnnClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class MatmulStrategy:
    def __init__(self):
        self.fake_config = SparseMatrixTestConfiguration(0, 0, 0, 0, 0, 0, 0)
        self.matmul_strategy = None
        self.thread_num = -1

    def set_matmul_strategy(self, matmul_strategy):
        self.matmul_strategy = matmul_strategy

    def set_thread_num(self, num):
        self.thread_num = num

    def matmul(self, x, y):
        assert self.matmul_strategy is not None, 'Please set matmul strategy first.'
        if self.thread_num != -1:
            return self.matmul_strategy(x, y, self.fake_config, self.thread_num)
        return self.matmul_strategy(x, y, self.fake_config)

class Logger:
    def __init__(self):
        self.start_time = dict()
        self.end_time = dict()
    
    def reset(self):
        self.start_time.clear()
        self.end_time.clear()
    
    def start(self, name):
        self.start_time[name] = time.time()
    
    def end(self, name):
        self.end_time[name] = time.time()

    def get_total(self):
        total = 0
        for name in self.start_time:
            total += self.end_time[name] - self.start_time[name]
        return total

    def show(self):
        total = 0
        for name in self.start_time:
            # print(f'{name}: {self.end_time[name] - self.start_time[name]}')
            total += self.end_time[name] - self.start_time[name]
        print(f'Total: {total*1000:.2f}ms')

class SparseDnnClassifier(object):
    def __init__(self):
        self.matmul_strategy: MatmulStrategy = None
        self.logger = Logger()

    def load_state_dict(self, path):
        state_dict = torch.load(path)
        self.fc1_weight = state_dict['fc1.weight'].transpose(0, 1).to_sparse().to(device=DEVICE)
        display_tensor_as_image(state_dict['fc1.weight'].transpose(0, 1), 'mnist/weight/fc1_weight.png')
        self.fc1_bias = state_dict['fc1.bias']
        self.fc2_weight = state_dict['fc2.weight'].transpose(0, 1).to_sparse().to(device=DEVICE)
        display_tensor_as_image(state_dict['fc2.weight'].transpose(0, 1), 'mnist/weight/fc2_weight.png')
        self.fc2_bias = state_dict['fc2.bias']
        self.fc3_weight = state_dict['fc3.weight'].transpose(0, 1).to_sparse().to(device=DEVICE)
        display_tensor_as_image(state_dict['fc3.weight'].transpose(0, 1), 'mnist/weight/fc3_weight.png')
        self.fc3_bias = state_dict['fc3.bias']
        self.fc4_weight = state_dict['fc4.weight'].transpose(0, 1).to_sparse().to(device=DEVICE)
        display_tensor_as_image(state_dict['fc4.weight'].transpose(0, 1), 'mnist/weight/fc4_weight.png')
        self.fc4_bias = state_dict['fc4.bias']
        self.fc5_weight = state_dict['fc5.weight'].transpose(0, 1).to_sparse().to(device=DEVICE)
        display_tensor_as_image(state_dict['fc5.weight'].transpose(0, 1), 'mnist/weight/fc5_weight.png')
        self.fc5_bias = state_dict['fc5.bias']
        self.fc6_weight = state_dict['fc6.weight'].transpose(0, 1).to_sparse().to(device=DEVICE)
        display_tensor_as_image(state_dict['fc6.weight'].transpose(0, 1), 'mnist/weight/fc6_weight.png')
        self.fc6_bias = state_dict['fc6.bias']

    def forward_layer(self, index, x, weight, bias, relu=True):
        self.logger.start(f'fc{index}_matmul')
        x = self.matmul_strategy.matmul(x, weight)
        self.logger.end(f'fc{index}_matmul')

        x = x.to_dense()

        self.logger.start(f'fc{index}_bias')
        x = x + bias
        self.logger.end(f'fc{index}_bias')

        if relu:
            self.logger.start(f'fc{index}_relu')
            x = torch.relu(x)
            self.logger.end(f'fc{index}_relu')
        return x.to_sparse()

    def forward(self, x):
        assert self.matmul_strategy is not None, 'Please set matmul strategy first.'
        x = x.view(x.size(0), -1)
        x = x.to_sparse()
        x = self.forward_layer(0, x, self.fc1_weight, self.fc1_bias)
        x = self.forward_layer(1, x, self.fc2_weight, self.fc2_bias)
        x = self.forward_layer(2, x, self.fc3_weight, self.fc3_bias)
        x = self.forward_layer(3, x, self.fc4_weight, self.fc4_bias)
        x = self.forward_layer(4, x, self.fc5_weight, self.fc5_bias)
        x = self.forward_layer(5, x, self.fc6_weight, self.fc6_bias, relu=False)
        return x.to_dense()

def display_tensor_as_image(tensor, path):
    binary_tensor = tensor != 0
    plt.imshow(binary_tensor, cmap='gray')
    plt.savefig(path, dpi=600)

def prune_weights_and_biases(module, density):
    prune.l1_unstructured(module, name="weight", amount=1-density)
    prune.l1_unstructured(module, name="bias", amount=1-density)
    prune.remove(module, 'weight')
    prune.remove(module, 'bias')

def prune_model(model, density):
    for module in model.children():
        prune_weights_and_biases(module, density)

def pruning(density):
    # Load the model for inference
    model = DnnClassifier()
    model.load_state_dict(torch.load(FULL_MODEL_PATH))
    prune_model(model, density)
    torch.save(model.state_dict(), PRUNED_MODEL_PATH)

def train():
    model = DnnClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(EPOCH):
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
    torch.save(model.state_dict(), FULL_MODEL_PATH)

def _inference(model, strategy, thread_num):
    model.matmul_strategy.set_matmul_strategy(strategy)
    model.matmul_strategy.set_thread_num(thread_num)
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            model.logger.reset()
            outputs = model.forward(images)
            total += model.logger.get_total()
    return total

def inference_sparse():
    model = SparseDnnClassifier()
    model.matmul_strategy = MatmulStrategy()

    strategy_set = [
        ('builtin', ExtensionHandler.sparse_mm),
        ('parallel_structure', ExtensionHandler.parallel_structure_sparse_mm),
        ('openmp', ExtensionHandler.openmp_sparse_mm),
        ('std_thread', ExtensionHandler.std_thread_sparse_mm)
    ]
    threads_set = [1, 2, 4, 8, 16]
    density_set = [0.1, 0.2, 0.4, 0.8]
    
    for density in density_set:
        pruning(density)
        model.load_state_dict(PRUNED_MODEL_PATH)
        for thread_num in threads_set:
            output_string = f'density=[ {density} ], num_threads= [ {thread_num} ]\n'
            output_string += f'strategy, duration(ms)\n'
            output_string += "-" * 50 + "\n"
            for strategy_name, strategy in strategy_set:
                if strategy_name == 'builtin' or strategy_name == 'parallel_structure':
                    t = _inference(model, strategy, -1)
                else:
                    t = _inference(model, strategy, thread_num)
                output_string += f'{strategy_name},{t*1000:.3f}\n'
            output_string += "-" * 50 + "\n"
            print(output_string)

def show_weight():
    model = SparseDnnClassifier()
    pruning(0.1)
    model.load_state_dict(PRUNED_MODEL_PATH)

if __name__ == '__main__':
    train()
    inference_sparse()
    # show_weight()