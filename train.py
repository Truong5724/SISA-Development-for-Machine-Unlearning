import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sharded import sizeOfShard, getShardHash, fetchShardBatch, fetchValBatch
import os
from glob import glob
from time import time
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", default="purchase", help="Architecture to use, default purchase"
)

parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Train for the specified number of epochs, default 20",
)
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Size of the batches, relevant for both train and test, default 16",
)

parser.add_argument(
    "--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001"
)

parser.add_argument(
    "--dropout_rate",
    default=0.4,
    type=float,
    help="Dropout rate, if relevant, default 0.4",
)

parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Location of the datasetfile, default datasets/purchase/datasetfile",
)

parser.add_argument("--optimizer", default="sgd", help="Optimizer, default sgd")
parser.add_argument("--container", help="Name of the container")
parser.add_argument("--shards", default=1, type=int, help="Number of shards to train on, default 1")
parser.add_argument(
    "--slices", default=1, type=int, help="Number of slices to use, default 1"
)

parser.add_argument(
    "--chkpt_interval",
    default=1,
    type=int,
    help="Interval (in epochs) between two chkpts, -1 to disable chackpointing, default 1",
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

def cosine_center_loss(embeddings, center):
    center = F.normalize(center, dim=0)
    cos_sim = F.cosine_similarity(embeddings, center.unsqueeze(0), dim=1)
    loss = (1 - cos_sim).mean()
    return loss

centers = {}

for shard in tqdm(range(args.shards)):         
    shard_size = sizeOfShard(args.container, shard)
    slice_size = shard_size // args.slices
    avg_epochs_per_slice = (
        2 * args.slices / (args.slices + 1) * args.epochs / args.slices
    )
    loaded = False

    # We assume that shard n contain data of class n
    class_id = shard

    # Instantiate optimizer
    if args.optimizer == "adam":
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    else:
        raise "Unsupported optimizer"
    
    # Init scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs)

    # Init center vector
    center = torch.zeros(128).to(device)

    # Each shard we can get a different threshold, so we store them in a list to be able to use them during inference.
    threshold = 0.0

    for sl in tqdm(range(args.slices)):
        # Get slice hash using sharded lib.
        slice_hash = getShardHash(
            args.container, shard, until=(sl + 1) * slice_size
        )

        # If checkpoints exists, skip the slice.
        if not os.path.exists(
            "containers/{}/cache/{}.pt".format(args.container, slice_hash)
        ):
            # Initialize state.
            elapsed_time = 0
            start_epoch = 0
            slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(
                sl * avg_epochs_per_slice
            )

            # If weights are already in memory (from previous slice), skip loading.
            if not loaded:
                # Look for a recovery checkpoint for the slice.
                recovery_list = glob(
                    "containers/{}/cache/{}_*.pt".format(args.container, slice_hash)
                )
                if len(recovery_list) > 0:
                    print(
                        "Recovery mode for shard {} on slice {}".format(args.shard, sl)
                    )

                    # Load weights.
                    model.load_state_dict(torch.load(recovery_list[0]))
                    start_epoch = int(
                        recovery_list[0].split("/")[-1].split(".")[0].split("_")[1]
                    )

                    # Load time
                    with open(
                        "containers/{}/times/{}_{}.time".format(
                            args.container, slice_hash, start_epoch
                        ),
                        "r",
                    ) as f:
                        elapsed_time = float(f.read())

                # If there is no recovery checkpoint and this slice is not the first, load previous slice.
                elif sl > 0:
                    previous_slice_hash = getShardHash(
                        args.container, args.shard, until=sl * slice_size
                    )

                    # Load weights.
                    model.load_state_dict(
                        torch.load(
                            "containers/{}/cache/{}.pt".format(
                                args.container, previous_slice_hash
                            )
                        )
                    )

                # Mark model as loaded for next slices.
                loaded = True
            
            ### CẦN TỐI ƯU
            # If this is the first slice, no need to load anything.
            elif sl == 0:
                loaded = True

            # Actual training.
            train_time = 0.0

            for epoch in tqdm(range(start_epoch, slice_epochs)):
                model.train()

                epoch_start_time = time()
                running_loss = 0.0
                all_embeddings = []

                for images, labels in fetchShardBatch(
                    args.container,
                    shard,
                    args.batch_size,
                    args.dataset,
                    class_id,
                    until=(sl + 1) * slice_size if sl < args.slices - 1 else None,
                ):
                    # Convert data to torch format and send to selected device.
                    gpu_images = torch.from_numpy(images).to(device)
                    gpu_labels = torch.from_numpy(labels).to(device)  # pylint: disable=no-member

                    forward_start_time = time()

                    # Perform basic training step.
                    embeddings = model(gpu_images)
                    loss = cosine_center_loss(embeddings, center)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_time += time() - forward_start_time
                    running_loss += loss.item()
                    all_embeddings.append(embeddings.detach())
                
                # ===== UPDATE CENTER =====
                all_embeddings = torch.cat(all_embeddings)
                center = all_embeddings.mean(dim=0)

                # ===== THRESHOLD =====
                cos_vals = F.cosine_similarity(all_embeddings, center.unsqueeze(0), dim=1)
                threshold = cos_vals.mean() - 2 * cos_vals.std()

                # ===== VALIDATION =====
                model.eval()   

                correct = 0
                total = 0

                with torch.no_grad():  
                    for val_images, val_labels in fetchValBatch(args.dataset, args.batch_size):
                        gpu_val_images = torch.from_numpy(val_images).to(device)
                        gpu_val_labels = torch.from_numpy(val_labels).to(device)

                        binary_labels = (gpu_val_labels == class_id).long()

                        val_embeddings = model(gpu_val_images)
                        cos_sim = F.cosine_similarity(val_embeddings, center.unsqueeze(0), dim=1)

                        preds = (cos_sim > threshold).long()

                        correct += (preds == binary_labels).sum().item()
                        total += binary_labels.size(0)

                val_acc = 100 * correct / total
                print(f" [Epoch {epoch+1}] - Loss: {running_loss:.4f} - Val accuracy : {val_acc:.2f}%")

                scheduler.step()

                # Create a checkpoint every chkpt_interval.
                if (
                    args.chkpt_interval != -1
                    and epoch % args.chkpt_interval == args.chkpt_interval - 1
                ):
                    # Save weights
                    torch.save(
                        model.state_dict(),
                        "containers/{}/cache/{}_{}.pt".format(
                            args.container, slice_hash, epoch
                        ),
                    )

                    # Save time
                    with open(
                        "containers/{}/times/{}_{}.time".format(
                            args.container, slice_hash, epoch
                        ),
                        "w",
                    ) as f:
                        f.write("{}\n".format(train_time + elapsed_time))

                    # Remove previous checkpoint.
                    if os.path.exists(
                        "containers/{}/cache/{}_{}.pt".format(
                            args.container, slice_hash, epoch - args.chkpt_interval
                        )
                    ):
                        os.remove(
                            "containers/{}/cache/{}_{}.pt".format(
                                args.container, slice_hash, epoch - args.chkpt_interval
                            )
                        )
                    if os.path.exists(
                        "containers/{}/times/{}_{}.time".format(
                            args.container, slice_hash, epoch - args.chkpt_interval
                        )
                    ):
                        os.remove(
                            "containers/{}/times/{}_{}.time".format(
                                args.container, slice_hash, epoch - args.chkpt_interval
                            )
                        )

            # When training is complete, save slice.
            torch.save(
                model.state_dict(),
                "containers/{}/cache/{}.pt".format(args.container, slice_hash),
            )
            with open(
                "containers/{}/times/{}.time".format(args.container, slice_hash), "w"
            ) as f:
                f.write("{}\n".format(train_time + elapsed_time))

            ### CẦN TỐI ƯU
            # Remove previous checkpoint.
            if os.path.exists(
                "containers/{}/cache/{}_{}.pt".format(
                    args.container, slice_hash, args.epochs - args.chkpt_interval
                )
            ):
                os.remove(
                    "containers/{}/cache/{}_{}.pt".format(
                        args.container, slice_hash, args.epochs - args.chkpt_interval
                    )
                )

            if os.path.exists(
                "containers/{}/times/{}_{}.time".format(
                    args.container, slice_hash, args.epochs - args.chkpt_interval
                )
            ):
                os.remove(
                    "containers/{}/times/{}_{}.time".format(
                        args.container, slice_hash, args.epochs - args.chkpt_interval
                    )
                )

            # If this is the last slice, create a symlink attached to it.
            if sl == args.slices - 1:
                os.symlink(
                    "{}.pt".format(slice_hash),
                    "containers/{}/cache/shard-{}.pt".format(
                        args.container, args.shard
                    ),
                )
                os.symlink(
                    "{}.time".format(slice_hash),
                    "containers/{}/times/shard-{}.time".format(
                        args.container, args.shard
                    ),
                )

        elif sl == args.slices - 1:
            os.symlink(
                "{}.pt".format(slice_hash),
                "containers/{}/cache/shard-{}.pt".format(
                    args.container, args.shard
                ),
            )
            if not os.path.exists(
                "containers/{}/times/shard-{}.time".format(
                    args.container, args.shard
                )
            ):
                os.symlink(
                    "null.time",
                    "containers/{}/times/shard-{}.time".format(
                        args.container, args.shard
                    ),
                )

    centers[str(shard)] = {
        "center": center.cpu().numpy().tolist(),
        "threshold": threshold
    }

# Save thresholds for inference use.
with open("containers/{}/centers.json".format(args.container), "w") as f:
    json.dump(centers, f)