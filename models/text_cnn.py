import torch.nn as nn
import torch
import torch.nn.functional as F
from .dynamic_conv import DynamicConv1d

class DynamicTextCNN(nn.Module):
    def __init__(self, input_dim, num_filters, filter_sizes, K=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            DynamicConv1d(input_dim, num_filters, k, K)
            for k in filter_sizes
        ])
        self.layer_norm = nn.LayerNorm(len(filter_sizes) * num_filters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        convs = [F.relu(conv(x)) for conv in self.convs]

        pools = [F.adaptive_max_pool1d(c, 1).squeeze(-1) for c in convs]
      
        features = torch.cat(pools, dim=1)
        features = self.layer_norm(features)

        return self.dropout(features)