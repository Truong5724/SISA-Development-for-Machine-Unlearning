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

# Build partition: each shard contains indices for a single class
data_distribution = datasetfile["nb_data_per_shard"]
partitions = []
start = 0

for shard in range(shards):
    partitions.append(np.arange(start, start + data_distribution[str(shard)]))
    start += data_distribution[str(shard)]

# Save splitfile
split_path = os.path.join(container_dir, "splitfile.npy")
np.save(split_path, partitions)
print(f"Created splitfile with {len(partitions)} shards at {split_path}")

