#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

python train.py --model cifar10 --shards 10 --slices 1 --dataset datasets/cifar10/datasetfile --epochs 50 --batch_size 128 --learning_rate 0.1 --dropout_rate 0.2 --optimizer "sgd" --chkpt_interval 1 --container "cifar10" 
    