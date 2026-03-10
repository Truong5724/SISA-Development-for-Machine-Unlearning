import numpy as np
import torch
from sharded import fetchTestBatch
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

parser.add_argument(
    "--unlearn_shards",
    nargs="*",
    type=int,
    default=[],
    help="List of shard IDs to ignore during inference"
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
    for test_images, test_labels in fetchTestBatch(args.dataset, args.batch_size):
        # Filter data (images and labels) based on unlearn_shards.
        mask = ~np.isin(test_labels, args.unlearn_shards)

        test_images = test_images[mask]
        test_labels = test_labels[mask]

        if len(test_labels) == 0:
            continue

        gpu_test_images = torch.from_numpy(test_images).to(device)
        gpu_test_labels = torch.from_numpy(test_labels).to(device)

        scores = []

        for m in models:
            outputs = m(gpu_test_images)
            prob = torch.softmax(outputs, dim=1)[:, 1] # Softmax for class 1 - positive.
            scores.append(prob)

        score_matrix = torch.stack(scores, dim=1)

        max_prob, pred_class = torch.max(score_matrix, dim=1)

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


