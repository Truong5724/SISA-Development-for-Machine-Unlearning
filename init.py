import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--container", required=True, help="Name of the container to create")
parser.add_argument(
    "--shards",
    type=int,
    default=None,
    help="Number of shards to split the training data into (defaults to number of classes in dataset)"
)

parser.add_argument(
    "--dataset",
    default="datasets/cifar10/datasetfile",
    help="Path to the datasetfile JSON metadata"
)

args = parser.parse_args()

# Make sure container directory structure exists
container_dir = os.path.join("containers", args.container)
for sub in ("cache", "output", "times"):
    path = os.path.join(container_dir, sub)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

# Load dataset metadata and prepare splitfile
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())

# Assure number of shards equals number of classes 
nb_classes = datasetfile["nb_classes"]
shards = args.shards

if not shards == nb_classes:
    raise ValueError(f"Requested {shards} shards but dataset only has {nb_classes} classes")

# Build partition: each shard contains indices for a single class (for train and val)
def build_and_save_partitions(distribution, shards, save_path):
    partitions = []
    start = 0

    for shard in range(shards):
        size = distribution[str(shard)]
        partitions.append(np.arange(start, start + size))
        start += size

    np.save(save_path, partitions)
    print(f"Created splitfile with {len(partitions)} shards at {save_path}")

train_distribution = datasetfile["nb_train_data_per_shard"]
train_partitions = build_and_save_partitions(train_distribution, shards, "containers/train_splitfile.npy")

val_distribution = datasetfile["nb_val_data_per_shard"]
val_partitions = build_and_save_partitions(val_distribution, shards, "containers/val_splitfile.npy")
