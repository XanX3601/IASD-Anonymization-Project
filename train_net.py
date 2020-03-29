import argparse
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
parser.add_argument("--dataset", help="Index of dataset", required=True, choices=range(11), type=int)
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# Data
# --------------------
src.populate_datasets()
dataset = src.datasets[args.dataset]
index_train = int(dataset.x.shape[0] * 0.9)

x_train = torch.from_numpy(dataset.x[:index_train])
y_train = torch.from_numpy(dataset.y[:index_train])
x_test = torch.from_numpy(dataset.x[index_train:])
y_test = torch.from_numpy(dataset.y[index_train:])

dataset = src.Dataset(x_train, y_train)
data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Train
# --------------------

last_batch_index = len(dataset) // args.batch_size
if len(dataset) % args.batch_size == 0: last_batch_index -= 1

for epoch in range(args.epochs):

    model.train()

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
        t.set_description("Epoch {:03d} / {} - Batch loss = {:.9f}".format(epoch + 1, args.epochs, loss.item()))
        loss.backward()
        optimizer.step()

        if i_batch == last_batch_index:
            model.eval()
            with torch.no_grad():
                y = model(x_test)
                loss = loss_function(y, y_test)
                t.set_postfix_str("test_loss = {:.9f}".format(loss.item()))

    torch.save(model, args.path)
