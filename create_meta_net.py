import argparse

import torch

import src

parser = argparse.ArgumentParser(description="Script to create and save a neural network")
parser.add_argument("--path", help="Path of the model", required=True, type=str)
parser.add_argument("--meta", help="Input size vector", required=True, type=str)
parser.add_argument("--cuda", help="Using GPU or not", action="store_true")

# Parse arguments
# --------------------
args = parser.parse_args()

# Using CUDA is asked and available
# --------------------
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Creating and saving the neural network
# --------------------
model = src.Neural_Network_Meta_Classifier(int(args.meta)).to(device)

torch.save(model, args.path)
