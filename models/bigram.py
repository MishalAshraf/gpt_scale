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

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super(Bigram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        

    
    def forward(self, x,targets=None):
        x = self.embedding(x)
        if targets==None:
            loss = None
        else:
            B,T,C = x.shape
            x = x.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(x,targets)

        return x, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is a B,T array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)

        return idx
    