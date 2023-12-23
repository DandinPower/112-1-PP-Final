from torch.nn.utils import prune
import torch.nn as nn

def prune_weights_and_biases(module: nn.Module, density: float):
    prune.l1_unstructured(module, name="weight", amount=1-density)
    prune.l1_unstructured(module, name="bias", amount=1-density)
    prune.remove(module, 'weight')
    prune.remove(module, 'bias')

def prune_model(model: nn.Module, density: float):
    for module in model.children():
        prune_weights_and_biases(module, density)