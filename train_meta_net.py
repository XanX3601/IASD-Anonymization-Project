import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

import src

parser = argparse.ArgumentParser(description="Script to train a neural network")

parser.add_argument("--epochs", help="Number of epochs", default=30, type=int)
parser.add_argument("--batch-size", help="Batch size", default=32, type=int)
parser.add_argument("--path", help="Path of the model", required=True, type=str)
parser.add_argument("--dataset", help="Directory of meta dataset", required=True, type=str)
parser.add_argument("--cuda", help="Using GPU or not", action="store_true")

# Parse arguments
# --------------------
args = parser.parse_args()

# Using CUDA is asked and available
# --------------------
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load model
# --------------------
model = torch.load(args.path).to(device)

# Loss and optimizer
# --------------------
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Data
# --------------------
x = torch.load(os.path.join(args.dataset, "x_meta.pt"))
y = torch.load(os.path.join(args.dataset, "y_meta.pt"))

dataset = src.Dataset(x, y)
data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Train
# --------------------

last_batch_index = len(dataset) // args.batch_size
if len(dataset) % args.batch_size == 0: last_batch_index -= 1

for epoch in range(args.epochs):

    model.train()
    running_loss = 0.0

    t = tqdm(
        enumerate(data_loader),
        desc="Epoch {:03d} / {} - Batch loss = ---".format(epoch + 1, args.epochs),
        total=len(dataset) // args.batch_size,
        unit="batch",
    )
    for i_batch, batch in t:
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        running_loss += loss.item()
        t.set_description("Epoch {:03d} / {} - Batch loss = {:.9f}".format(epoch + 1, args.epochs, loss.item()))
        loss.backward()
        optimizer.step()

        if i_batch == last_batch_index:
            t.set_postfix_str("BCELoss = {:.9f}".format(running_loss / (last_batch_index + 1)))

    torch.save(model, args.path)
