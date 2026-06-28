#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ ! -f cifar10-general-report.csv ]]; then
    echo "nb_shards, nb_slices, forgot_shard_id, retained_accuracy, unlearning_accuracy, retained_precision_macro, retained_recall_macro, retained_f1_macro, training_time" > cifar10-general-report.csv
fi

slices=1

### Inference without unlearning.
metric=$(python inference.py \
    --model cifar10 \
    --shards 10 \
    --dataset datasets/cifar10/datasetfile \
    --batch_size 64 \
    --dropout_rate 0.3 \
    --container "cifar10" | tail -n 1)

cat containers/cifar10/times/shard-*.time > "containers/cifar10/times/times.tmp"
time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
python class_stats.py --container "cifar10"

### Append to csv file.
echo "10,${slices},None,${metric},${time}" >> cifar10-general-report.csv

### Unlearning shards
cases=(
    "0"
    "3"
    "5"
    "0 3 8"
    "1 6 9"
    "2 4 7"
    "0 3 5 8 9"
)

for case in "${cases[@]}"
do
    # Convert string to array
    IFS=' ' read -ra forgot_shards <<< "$case"

    echo "Unlearning shards: ${forgot_shards[*]}"

    metric=$(python inference.py \
        --model cifar10 \
        --shards 10 \
        --dataset datasets/cifar10/datasetfile \
        --batch_size 64 \
        --dropout_rate 0.2 \
        --container "cifar10" \
        --unlearn_shards "${forgot_shards[@]}" | tail -n 1)

    cat containers/cifar10/times/shard-*.time > "containers/cifar10/times/times.tmp"
    time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
    python class_stats.py --container "cifar10" --unlearn_shards "${forgot_shards[@]}"

    echo "10,${slices},\"[$(IFS=,; echo "${forgot_shards[*]}")]\",${metric},${time}" >> cifar10-general-report.csv
done