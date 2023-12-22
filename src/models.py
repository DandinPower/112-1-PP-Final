from src.utils import Logger
from src.matmul_strategy import MatmulStrategy
import torch.nn as nn
import torch 

# because we are not using GPU strategy, we need to set the device to CPU
DEVICE = torch.device('cpu')

class DnnClassifier(nn.Module):
    def __init__(self):
        super(DnnClassifier, self).__init__()
        self.fc0 = nn.Linear(784, 4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class SparseLayer:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, is_relu=True):
        self.weight = weight
        self.bias = bias
        self.is_relu = is_relu

class SparseDnnClassifier:
    """
    A class that represents a sparse neural network.
    You must first load the state dict of the model, and then set the matmul strategy.
    Please make sure that the model you want to load is a ``DnnClassifier``.
    After that, you can use the ``forward`` function to do inference.
    """
    def __init__(self):
        self.matmul_strategy: MatmulStrategy = None
        self.logger = Logger()
        self.layers:list[SparseLayer] = []
        self.num_layers = 0

    def add_layer(self, state_dict:dict, weight_name: str, bias_name: str, is_relu=True):
        weight = state_dict[weight_name].transpose(0, 1).to_sparse().to(device=DEVICE)
        bias = state_dict[bias_name]
        self.layers.append(SparseLayer(weight, bias, is_relu))

    def load_state_dict(self, path: str, num_layers: int):
        self.layers.clear()
        self.num_layers = num_layers
        state_dict = torch.load(path)
        for i in range(num_layers):
            # the last layer does not have relu
            if i == num_layers - 1:
                self.add_layer(state_dict, f'fc{i}.weight', f'fc{i}.bias', is_relu=False)
            else: 
                self.add_layer(state_dict, f'fc{i}.weight', f'fc{i}.bias')

    def forward_layer(self, input: torch.Tensor, layer_index: int):
        # matmul
        self.logger.start(f'fc{layer_index}_matmul')
        output = self.matmul_strategy.matmul(input, self.layers[layer_index].weight)
        self.logger.end(f'fc{layer_index}_matmul')

        # add bias
        output = output.to_dense()
        self.logger.start(f'fc{layer_index}_bias')
        output = output + self.layers[layer_index].bias
        self.logger.end(f'fc{layer_index}_bias')

        # relu
        if self.layers[layer_index].is_relu:
            self.logger.start(f'fc{layer_index}_relu')
            output = torch.relu(output)
            self.logger.end(f'fc{layer_index}_relu')
        return output.to_sparse()

    def forward(self, x: torch.Tensor):
        assert self.matmul_strategy is not None, 'Please set matmul strategy first.'
        assert self.num_layers != 0, 'Please load state dict first.'
        x = x.view(x.size(0), -1)
        x = x.to_sparse()
        for i in range(self.num_layers):
            x = self.forward_layer(x, i)
        return x.to_dense()