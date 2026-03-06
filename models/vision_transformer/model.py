import torch
import torch.nn as nn
from .dataset import FlowersDataset
from primitive_modules.layernorm import MyLayerNorm
import math


class InputEmbeddings(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int, height:int, width: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, patch_size)
        self.pos_embedd = nn.Parameter(torch.randn(1, (height*width) // (patch_size*patch_size) + 1, d_model)) # +1 for [cls_token]
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        inputs = x + self.pos_embedd
        return inputs
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # (B, seq_len, d_model) -> (B, n_heads, seq_len, d_k)  dk = d_model / n_heads
        b, seq_len, d_model = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(b, seq_len, self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.view(b, seq_len, self.n_heads, -1).permute(0, 2, 3, 1) #k transpose in fact
        v = v.view(b, seq_len, self.n_heads, -1).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k) / math.sqrt(self.d_k)
        if mask is not None:

            if mask.dim() == 2:
                 mask = mask.unsqueeze(1).unsqueeze(2)
            
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_score = torch.softmax(scores, dim = -1) @ v
        attention_score = attention_score.permute(0, 2, 1, 3).contiguous().reshape(b, seq_len, -1)
        output = self.w_o(attention_score)

        return self.dropout(output)
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion_scale: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model*expansion_scale)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(d_model*expansion_scale, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int, dropout: float, expansion_scale: int) :
        super().__init__()
        self.mha = MultiHeadAttention(d_model, seq_len, n_heads, dropout)
        self.ffn = FeedForward(d_model, expansion_scale, dropout)
        self.ln1 = MyLayerNorm((d_model,))
        self.ln2 = MyLayerNorm((d_model,))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask = None):
        identity = x
        x = self.ln1(x)
        x = self.mha(x, mask)
        x = self.dropout1(x)
        x = x + identity
        identity = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + identity

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, patch_size, d_model: int, seq_len: int, n_heads: int, dropout: float, expansion_scale: int, n_layers: int):
        super().__init__()
        self.input_emb = InputEmbeddings(in_channels, patch_size, d_model, height, width)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, seq_len, n_heads, dropout, expansion_scale) for _ in range(n_layers)])

        self.ln_f = MyLayerNorm((d_model,))

    def forward(self, x, mask = None):
        x = self.input_emb(x)

        for layer in self.encoder_blocks:
            x = layer(x, mask)

        return self.ln_f(x)
    
class VisionTransformerClassifier(nn.Module):
    def __init__(self, n_classes: int, in_channels: int, height: int, width: int, patch_size, d_model: int, seq_len: int, n_heads: int, dropout: float, expansion_scale: int, n_layers: int):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(in_channels, height, width, patch_size, d_model, seq_len, n_heads, dropout, expansion_scale, n_layers)
        self.linear1 = nn.Linear(d_model, n_classes)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(n_classes, n_classes)

    def forward(self, x):
        
        x = self.transformer_encoder(x)
        x = x[:, 0, :]

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x



# vit = VisionTransformerClassifier(n_classes=3, in_channels=3, height=224, width=224, patch_size=16, d_model=384, seq_len=197, n_heads=4, dropout=0.1, expansion_scale=4, n_layers=6)
# img = torch.randn(1, 3, 224, 224)
# out = vit(img)
# print(out)