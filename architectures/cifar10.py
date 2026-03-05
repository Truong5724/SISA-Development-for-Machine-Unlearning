import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class Model(nn.Module):
    def __init__(self, embedding_dim=128, dropout_rate=0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),

            conv_block(64, 128),
            conv_block(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout_rate)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return F.normalize(x, dim=1)