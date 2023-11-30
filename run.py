import numpy as np
import pandas as pd
import os
import sys
import time
import pickle
import argparse
import torch
from models.bigram import Bigram
from models.transformer import Transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_embd', type=int)#, default=64)
    parser.add_argument('--dropout', type=float)#, default=0.2)
    parser.add_argument('--num_heads', type=int)#, default=8)
    args = parser.parse_args()

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))

    vocab_size = len(chars)
    batch_size = 64 # how many independent sequences will we process in parallel?

    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 3000
    eval_interval = 500
    eval_iters = 100
    lr = 3e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = {}
    params['n_embd'] = 64
    params['dropout'] = 0.2
    params['num_heads'] = 8

    if args.n_embd is not None:
        params['n_embd'] = args.n_embd
    if args.dropout is not None:
        params['dropout'] = args.dropout
    if args.num_heads is not None:
        params['num_heads'] = args.num_heads

    ### graphs look like, change n_embd do same number of runs, cycle thru a range of values, fix heads. 3
    ### graphs, one for each of the 3 params, n_embd, dropout, num_heads

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]


    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        if split == 'train':
            data = train_data
        else: 
            data = val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train','val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X,Y = get_batch(split)
                logits, loss = model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            model.train()
        return out
        

    #model = Bigram(vocab_size)

    model = Transformer(
                        vocab_size, params['n_embd'],
                        num_heads=params['num_heads'],
                        block_size=block_size,dropout=params['dropout'])

    m = model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    result_list = []

    now = time.time()
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            result_dict = {
                    'iter': iter,
                    'train_loss': losses['train'].item(),
                    'val_loss': losses['val'].item(),
                    'sample': decode(model.generate(torch.randint(vocab_size, (1, 1)), 100).squeeze().tolist()),
                    'time': time.time() - now
            }
            print(f'iter {iter:5d} | train loss {losses["train"]:5.2f} | val loss {losses["val"]:5.2f}')
            print(f"sample: {decode(model.generate(torch.randint(vocab_size, (1, 1)), 100).squeeze().tolist())}")

            result_list.append(result_dict)

        xb,yb = get_batch('train')

        logits, loss = model(xb,yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    df_results = pd.DataFrame(result_list)  
    df_results.to_csv(f"n_embd_{str(params['n_embd'])}_results.csv", index=False)