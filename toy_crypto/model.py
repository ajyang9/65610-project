import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.input(x))
        for block in self.blocks:
            x = self.act(block(x))
        x = self.output(x)        
        return x
