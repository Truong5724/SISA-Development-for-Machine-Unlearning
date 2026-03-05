#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

python train.py --model cifar10 --shards 10 --slices 1 --dataset datasets/cifar10/datasetfile --epochs 100 --batch_size 32 --learning_rate 0.05 --dropout_rate 0.2 --optimizer adam --chkpt_interval 1 --container "cifar10" 
    