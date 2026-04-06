#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ ! -f cifar10-general-report.csv ]]; then
    echo "nb_shards, nb_slices, forgot_shard_id, retained_accuracy, unlearning_accuracy, retained_precision_macro, retained_recall_macro, retained_f1_macro, training_time" > cifar10-general-report.csv
fi

slices=3

### Uncomment the case depending on the experiment you want to run.

### Inference without unlearning.
metric=$(python inference.py --model cifar10 --shards 10 --slices ${slices} --dataset datasets/cifar10/datasetfile --batch_size 64 --dropout_rate 0.3 --container "cifar10" | tail -n 1)


### Unlearning one shard at a time.
# forgot_shards=($(shuf -i 0-9 -n 1))
# metric=$(python inference.py --model cifar10 --shards 10 --slices ${slices} --dataset datasets/cifar10/datasetfile --batch_size 64 --dropout_rate 0.3 --container "cifar10" --unlearn_shards "${forgot_shards[@]}" | tail -n 1)


### Unlearning n shards at a time (Modify `n` to meet your needs).
# forgot_shards=($(shuf -i 0-9 -n 4))
# metric=$(python inference.py --model cifar10 --shards 10 --slices ${slices} --dataset datasets/cifar10/datasetfile --batch_size 64 --dropout_rate 0.3 --container "cifar10" --unlearn_shards "${forgot_shards[@]}" | tail -n 1)


cat containers/cifar10/times/shard-*.time > "containers/cifar10/times/times.tmp"
time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
python class_stats.py --container "cifar10"


### Append to csv file 
# Use for no unlearning case
echo "10,${slices},None,${metric},${time}" >> cifar10-general-report.csv

# Use for unlearning case
# echo "10,${slices},\"[$(IFS=,; echo "${forgot_shards[*]}")]\",${metric},${time}" >> cifar10-general-report.csv