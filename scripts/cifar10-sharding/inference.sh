#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

python inference.py --model cifar10 --shards 10 --slices 1 --dataset datasets/cifar10/datasetfile --batch_size 32 --dropout_rate 0.2 --container "cifar10" 
