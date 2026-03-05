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

train_map = create_label_map(X_train, y_train)

if not os.path.exists(f'cifar10_train.npy'):
    np.save(f'cifar10_train.npy', train_map, allow_pickle=True)

if not os.path.exists(f'cifar10_val.npy'):
    np.save(f'cifar10_val.npy', {'X': X_val, 'y': y_val})

if not os.path.exists(f'cifar10_test.npy'):
    np.save(f'cifar10_test.npy', {'X': test_images, 'y': np.array(test_labels)})

# Mapping label to class name
with open('cifar-10-batches-py/batches.meta', 'rb') as f:
    meta = pickle.load(f, encoding='bytes')
    label_names = [name.decode('utf-8') for name in meta[b'label_names']]
    label_map = {i: name for i, name in enumerate(label_names)}

# Distribution of data per label in the training set
nb_data_per_label = {str(label) : len(images) for label, images in train_map.items()}

# Update datasetfile (metadata)
if not os.path.exists("datasetfile"):
    dataset_info = {
        "nb_train": len(X_train),
        "nb_val": len(X_val),
        "nb_test": len(test_images),
        "input_shape": train_images.shape[1:],
        "nb_classes": 10,
        "dataloader": "dataloader",
        "label_map": label_map,
        "nb_data_per_label": nb_data_per_label
    }

    with open("datasetfile", "w") as f:
        json.dump(dataset_info, f, indent=4)