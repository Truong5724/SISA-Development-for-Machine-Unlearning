#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

python train.py --model cifar10 --shards 10 --slices 3 --dataset datasets/cifar10/datasetfile --epochs 100 --batch_size 64 --learning_rate 0.05 --dropout_rate 0.3 --optimizer "sgd" --chkpt_interval 3 --container "cifar10" 
    