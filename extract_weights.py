import argparse
import os

import torch
from tqdm import tqdm

import src

parser = argparse.ArgumentParser(description="Script to extract weights from neural networks and create a dataset")
parser.add_argument("--dir", help="Directory of neural networks models", required=True, type=str)
parser.add_argument("--out", help="Directory of output tensors", required=True, type=str)
parser.add_argument("--cuda", help="Using GPU or not", action="store_true")

# Parse arguments
# --------------------
args = parser.parse_args()

# Using CUDA is asked and available
# --------------------
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Extracting weights
# --------------------
x, y = [], []
src.populate_datasets()

for net in tqdm(os.listdir(args.dir)):
    index = int(net.split("_")[-1].split(".")[0])
    model = torch.load(os.path.join(args.dir, net))
    tensor = next(model.dense_out.parameters()).to(device)
    x.append(tensor)
    y.append(torch.tensor([[float(src.datasets[index].label)]]))

# Creating the dataset
# --------------------
x_data = torch.cat(x, 0).to(device)
y_data = torch.cat(y, 0).to(device)

torch.save(x_data, os.path.join(args.out, "x_meta.pt"))
torch.save(y_data, os.path.join(args.out, "y_meta.pt"))
