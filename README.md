# SISA Development for Machine Unlearning

### Faculty of Information Technology – University of Science, VNU‑HCM

### Project Information
- **Supervisor:** Assoc. Prof. Le Hoang Thai
- **Students:**
  - 22120384 – Nguyen Dinh Tri
  - 22120398 – Vu Hoang Nhat Truong

### Acknowledgement
This graduation thesis builds on prior work in the rapidly growing field of
machine unlearning. We particularly acknowledge the seminal paper by
Bourtoule *et al.* (2021) which introduced many of the concepts and algorithms
that underpin our implementation.

> Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C., Jia, H., Travers, A., Zhang, B., Lie, D., & Papernot, N. (2021). *Machine Unlearning*. Proceedings of
the 42nd IEEE Symposium on Security and Privacy.

## Running the Experiment

Follow the steps below to run the pipeline.

### 1. Download the dataset
```bash
cd datasets/cifar10
python download.py
```

### 2. Prepare the data
```bash
python prepare_data.py
```

### 3. Create splitfile and container
```bash
# Run `cd ../..` first if you are not in the project root directory.
bash scripts/cifar10-sharding/init.sh
```

### 4. Train the model
```bash
bash scripts/cifar10-sharding/train.sh
```

### 5. Run inference
```bash
bash scripts/cifar10-sharding/inference.sh
``` 

**Note:** You can modify the input arguments inside the `.sh` scripts to customize the experiment settings.