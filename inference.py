import numpy as np
import torch
import torch.nn.functional as F

from sharded import fetchTestBatch

import json
from tqdm import tqdm
import argparse

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
model.to(device)

with open("containers/{}/centers.json".format(args.container)) as f:
    centers = json.loads(f.read())

for shard in tqdm(range(args.shards)):
    # Load model weights from shard checkpoint (last slice).
    model.load_state_dict(
        torch.load(
            "containers/{}/cache/shard-{}.pt".format(
                args.container, shard
            )
        )
    )

    class_id = shard
    correct = 0
    total = 0

    center = torch.from_numpy(np.array(centers[str(shard)]["center"])).to(device)
    threshold = centers[str(shard)]["threshold"]

    model.eval()
    with torch.no_grad():  
        for test_images, test_labels in fetchTestBatch(args.dataset, args.batch_size):
            gpu_test_images = torch.from_numpy(test_images).to(device)
            gpu_test_labels = torch.from_numpy(test_labels).to(device)

            binary_labels = (gpu_test_labels == class_id).long()

            test_embeddings = model(gpu_test_images)
            cos_sim = F.cosine_similarity(test_embeddings, center.unsqueeze(0), dim=1)

            preds = (cos_sim > threshold).long()

            correct += (preds == binary_labels).sum().item()
            total += binary_labels.size(0)

    test_acc = 100 * correct / total
    print(f" Shard{shard} accuracy : {test_acc:.2f}%")