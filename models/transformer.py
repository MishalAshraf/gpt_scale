import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import time
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader

## a transformer is a multi-head self-attention network

### head contains the attention mechanism 
### stacking multi-heads in parallel. 
### chain blocks of these together to get a transformer

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', 
                             torch.tril(torch.ones(block_size, 
                                                   block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5 #k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd,block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,n_embd, 
                                         block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), 
                                 nn.ReLU(),
                                 nn.Linear(4*n_embd, n_embd))
        
    def forward(self, x):
        return self.net(x)
    

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    
class Block(nn.Module):
    def __init__(self, n_embd, num_heads,block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.attn = MultiHeadAttention(num_heads, head_size, n_embd,
                                       block_size, dropout)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd, True)
        self.ln2 = LayerNorm(n_embd, True)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self,vocab_size,n_embd,num_heads,block_size,dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
                         Block(n_embd, num_heads, 
                               block_size=block_size, dropout=dropout),
                         Block(n_embd, num_heads, 
                               block_size=block_size, dropout=dropout),
                         Block(n_embd, num_heads, 
                               block_size=block_size, dropout=dropout),
                         nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, device='cpu'):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
    
    



