from numpy import empty_like
import torch
import torch.nn as nn


class MyDropout(nn.Module):
    def __init__(self, p) -> None:
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = torch.empty_like(x)
            mask.bernoulli_(1-self.p)
            x = x*mask / (1 - self.p)


        return x

