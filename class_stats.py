import numpy as np
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--container", help="Name of the container")

args = parser.parse_args()

# Load file
data = np.load(f"containers/{args.container}/output/predictions.npy")

# Split predictions and labels
all_preds = data[:, 0]
all_labels = data[:, 1]

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure()

sns.heatmap(cm, annot=True, fmt="d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.tight_layout()
plt.savefig(f"cm_{args.container}.png")

# Per-class metrics
precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

print("Per-class metrics:")
for i in range(len(precision)):
    print("="*10 + f"Class {i}" + "="*10)
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall:    {recall[i]:.4f}")
    print(f"F1-score:  {f1[i]:.4f}")

