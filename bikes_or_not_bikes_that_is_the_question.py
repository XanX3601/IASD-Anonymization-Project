import argparse

import torch

parser = argparse.ArgumentParser(description="Bikes or not bikes, that is the question")

parser.add_argument("--target", help="Path of the target model", required=True, type=str)
parser.add_argument("--meta", help="Path of the meta model", required=True, type=str)
parser.add_argument("--cuda", help="Using GPU or not", action="store_true")

# Parse arguments
# --------------------
args = parser.parse_args()

# Using CUDA is asked and available
# --------------------
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load models
# --------------------
target_model = torch.load(args.target)
meta_model = torch.load(args.meta)

# Extract weights from target
# --------------------
weights = next(target_model.dense_out.parameters()).to(device)
answer_to_life = meta_model(weights)
#print("Meta classifier output: {}".format(answer_to_life.item()))
#print("{}bikes".format("" if answer_to_life.item() > 0.5 else "no "))
print("{}".format("1" if answer_to_life.item() > 0.5 else "0"))
