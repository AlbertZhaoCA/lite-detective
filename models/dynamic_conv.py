import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, K=4, reduction=4):
        super().__init__()
        self.K = K
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding='same')
            for _ in range(K)
        ])
        # self.residual_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.SiLU(),
            nn.Conv1d(in_channels // reduction, in_channels // reduction, 1),
            nn.SiLU(),
            nn.Conv1d(in_channels // reduction, K, 1)
        )
        nn.init.normal_(self.attn[-1].weight, mean=0, std=0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        attn_logits = self.attn(x) 
        attn_weights = F.softmax(attn_logits, dim=1)
        conv_outs = [conv(x) for conv in self.convs]
        out = sum(w * o for w, o in zip(attn_weights.split(1, dim=1), conv_outs))
        # residual connection
        return out + x
