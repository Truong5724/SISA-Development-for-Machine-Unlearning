import os
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Read 1 batch, return (data, labels)
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'] # (10000, 3072)
        labels = batch[b'labels'] # 10000

        # Convert to (C, H, W) to suitable for PyTorch input
        images = images.reshape((10000, 3, 32, 32))
        return images, labels

train_images = []
train_labels = []

# Download train set
for i in range(1, 6):
    batch_images, batch_labels = load_cifar_batch(f"cifar-10-batches-py/data_batch_{i}")
    train_images.append(batch_images)
    train_labels.append(batch_labels)

train_images = np.concatenate(train_images, axis=0) # (50000, 3, 32, 32)
train_labels = np.concatenate(train_labels, axis=0) # (50000, )

# Download test set
test_images, test_labels = load_cifar_batch(f"cifar-10-batches-py/test_batch")

# Split train set into train and val sets (90% train, 10% val)
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, stratify=train_labels)

# Build and save a label-wise map for the training set
def create_label_map(X, y):
    """Return a dictionary mapping each label to an array of examples (in ascending order of labels)."""
    label_map = {}
    for img, lbl in zip(X, y):
        label_map.setdefault(lbl, []).append(img)

    # Convert lists to numpy arrays for efficiency
    for lbl in label_map:
        label_map[lbl] = np.array(label_map[lbl])

    # Return a new dict ordered by sorted labels
    sorted_map = {key: label_map[key] for key in sorted(label_map)}
    return sorted_map



# For each shard we want the number of positive and negative data are the same (not bias), either in train or test set.
# Consequently, val sets for each shard will likely be different with each others.
def build_binary_shards(label_map):
    np.random.seed(57)

    X_all = []
    y_all = []
    nb_data_per_shard = {}

    for shard_label in label_map.keys():

        true_images = label_map[shard_label]
        n_true = len(true_images)

        true_labels = np.ones(n_true)

        false_images = []
        per_other = n_true // (len(label_map) - 1)

        for other_label in label_map.keys():
            if other_label == shard_label:
                continue

            other_images = label_map[other_label]

            idx = np.random.choice(len(other_images), per_other, replace=False)
            sampled = other_images[idx]

            false_images.append(sampled)

        false_images = np.concatenate(false_images, axis=0)
        false_labels = np.zeros(len(false_images))

        X_shard = np.concatenate([true_images, false_images], axis=0)
        y_shard = np.concatenate([true_labels, false_labels], axis=0)

        perm = np.random.permutation(len(X_shard))
        X_shard = X_shard[perm]
        y_shard = y_shard[perm]

        nb_data_per_shard[str(shard_label)] = len(X_shard)

        X_all.append(X_shard)
        y_all.append(y_shard)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_all, y_all, nb_data_per_shard

# Save train
train_map = create_label_map(X_train, y_train)
X_train_all, y_train_all, nb_train_data_per_shard = build_binary_shards(train_map)

if not os.path.exists("cifar10_train.npy"):
    np.save("cifar10_train.npy", {'X': X_train_all, 'y': y_train_all})

# Save val
val_map = create_label_map(X_val, y_val)
X_val_all, y_val_all, nb_val_data_per_shard = build_binary_shards(val_map)

if not os.path.exists(f'cifar10_val.npy'):
    np.save(f'cifar10_val.npy', {'X': X_val_all, 'y': y_val_all})

# Save test
if not os.path.exists(f'cifar10_test.npy'):
    np.save(f'cifar10_test.npy', {'X': test_images, 'y': np.array(test_labels)})

# Mapping label to class name
with open('cifar-10-batches-py/batches.meta', 'rb') as f:
    meta = pickle.load(f, encoding='bytes')
    label_names = [name.decode('utf-8') for name in meta[b'label_names']]
    label_map = {str(i): name for i, name in enumerate(label_names)}

# Update datasetfile (metadata)
if not os.path.exists("datasetfile"):
    dataset_info = {
        "old_nb_train": len(X_train),
        "new_nb_train": len(X_train_all),
        "nb_train_data_per_shard": nb_train_data_per_shard,
        
        "old_nb_val": len(X_val),
        "new_nb_val": len(X_val_all),
        "nb_val_data_per_shard": nb_val_data_per_shard,

        "nb_test": len(test_images),
        "input_shape": train_images.shape[1:],
        "nb_classes": 10,
        "dataloader": "dataloader",
        "label_map": label_map,
    }

    with open("datasetfile", "w") as f:
        json.dump(dataset_info, f, indent=4)