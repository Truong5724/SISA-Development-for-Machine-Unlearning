import numpy as np
import torch
import torch.nn.functional as F

from sharded import fetchTestBatch

import json
from tqdm import tqdm
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="purchase", help="Architecture to use, default purchase"
)

parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Size of the batches, relevant for both train and test, default 16",
)

parser.add_argument(
    "--dropout_rate",
    default=0.4,
    type=float,
    help="Dropout rate, if relevant, default 0.4",
)

parser.add_argument("--container", help="Name of the container")
parser.add_argument("--shards", default=1, type=int, help="Number of shards to train on, default 1")

parser.add_argument(
    "--slices", default=1, type=int, help="Number of slices to use, default 1"
)
parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Location of the datasetfile, default datasets/purchase/datasetfile",
)

args = parser.parse_args()

# Import the architecture.
from importlib import import_module
model_lib = import_module("architectures.{}".format(args.model))

# Use GPU if available.
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # pylint: disable=no-member

model = model_lib.Model(dropout_rate=args.dropout_rate)

models = []

# Load model for each shard
for shard in range(args.shards):
    m = copy.deepcopy(model)
    m.load_state_dict(torch.load(f"containers/{args.container}/cache/shard-{shard}.pt"))
    m.to(device)
    m.eval()
    models.append(m)

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for test_images, test_labels in tqdm(fetchTestBatch(args.dataset, args.batch_size)):

        gpu_test_images = torch.from_numpy(test_images).to(device)
        gpu_test_labels = torch.from_numpy(test_labels).to(device)

        batch_preds = []

        for m in models:
            outputs = m(gpu_test_images)
            preds = torch.argmax(outputs, dim=1)
            batch_preds.append(preds)

        pred_matrix = torch.stack(batch_preds, dim=1)

        pred_class = torch.argmax(pred_matrix, dim=1)

        mask = pred_matrix.sum(dim=1) == 0
        pred_class[mask] = -1

        correct += (pred_class == gpu_test_labels).sum().item()
        total += gpu_test_labels.size(0)

        all_preds.append(pred_class.cpu().numpy())
        all_labels.append(gpu_test_labels.cpu().numpy())

# Save predictions and labels for analysis
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

output = np.stack((all_preds, all_labels), axis=1)
np.save(f"containers/{args.container}/output/predictions.npy", output)

acc = correct / total
print(acc)


