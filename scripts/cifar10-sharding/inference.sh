#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ ! -f cifar10-general-report.csv ]]; then
    echo "nb_shards, accuracy, retraining_time" > cifar10-general-report.csv
fi

acc=$(python inference.py --model cifar10 --shards 10 --slices 1 --dataset datasets/cifar10/datasetfile --batch_size 32 --dropout_rate 0.2 --container "cifar10")
cat containers/cifar10/times/shard-*.time > "containers/cifar10/times/times.tmp"
time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
echo "10,${acc},${time}" >> cifar10-general-report.csv 
