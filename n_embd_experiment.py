import subprocess
import os


for n in [8, 16, 32, 64, 128]:
    print(f"n_embd: {n}")
    subprocess.run(f"python3 run.py --n_embd {n}", shell=True)