import torch
import torch.nn as nn

class MyBatchNorm1d(nn.Module):
    def __init__(self, in_features: int, decay: float = 0.9, eps: float = 1e-5) -> None:
        super().__init__()
        self.decay = decay
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features)
        )
        self.register_buffer("running_mean", torch.zeros(in_features))
        self.register_buffer("running_var", torch.ones(in_features))
        
    def forward(self, x):

        if self.training:
            batch_mean = torch.mean(x, dim = 0)
            batch_var = torch.var(x, dim = 0, unbiased = False)
            batch_var_unbiased = torch.var(x, dim = 0)
            with torch.no_grad():
                self.running_mean = self.running_mean*self.decay + batch_mean * (1 - self.decay)
                self.running_var = self.running_var*self.decay + batch_var_unbiased * (1- self.decay)

        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        x_normalized = (x - batch_mean)/torch.sqrt(batch_var + self.eps)  
        out = self.gamma * x_normalized + self.beta
        
        return out


class MyBatchNorm2d(nn.Module):
    def __init__(self, in_features: int, decay: float = 0.9, eps: float = 1e-5) -> None:
        super().__init__()
        self.decay = decay
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features)
        )
        self.register_buffer("running_mean", torch.zeros(in_features))
        self.register_buffer("running_var", torch.ones(in_features))
        
    def forward(self, x):

        if self.training:
            batch_mean = torch.mean(x, dim = (0, 2,3),)
            batch_var = torch.var(x, dim = (0, 2, 3), unbiased = False)
            batch_var_unbiased = torch.var(x, dim = (0, 2, 3))
            with torch.no_grad():
                self.running_mean = self.running_mean*self.decay + batch_mean * (1 - self.decay)
                self.running_var = self.running_var*self.decay + batch_var_unbiased * (1- self.decay)

        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        batch_mean = batch_mean.view(1, -1, 1, 1)
        batch_var = batch_var.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        x_normalized = (x - batch_mean)/torch.sqrt(batch_var + self.eps)  
        out = gamma * x_normalized + beta
        
        return out
