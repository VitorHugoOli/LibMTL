# Create a base class to a model in pytorch
from torch import nn


class PCGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, dropout):
        super(PCGNN, self).__init__()
        pass

    def forward(self, x, edge_index):
        pass