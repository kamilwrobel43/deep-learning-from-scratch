import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5) -> None:
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        n_dims = x.ndim
        n_reduction_dims = len(self.normalized_shape)
        reduction_dim_list = list(range(n_dims - n_reduction_dims, n_dims))
        var = torch.var(x, correction = 0, dim = reduction_dim_list, keepdim = True)
        mean = torch.mean(x, dim = reduction_dim_list, keepdim = True)
    

        y = (x - mean) / torch.sqrt(var + self.eps)
        
        y = y*self.gamma + self.beta
        

        return y



