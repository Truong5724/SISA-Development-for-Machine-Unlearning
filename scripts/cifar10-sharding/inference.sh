#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ ! -f cifar10-general-report.csv ]]; then
    echo "nb_shards, forgot_shard_id, accuracy, retraining_time" > cifar10-general-report.csv
fi

### Uncomment the case depending on the experiment you want to run.

### Inference without unlearning.
acc=$(python inference.py --model cifar10 --shards 10 --slices 1 --dataset datasets/cifar10/datasetfile --batch_size 64 --dropout_rate 0.2 --container "cifar10" | tail -n 1)
cat containers/cifar10/times/shard-*.time > "containers/cifar10/times/times.tmp"
time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
echo "10,None,${acc},${time}" >> cifar10-general-report.csv

### Unlearning one shard at a time.

# for i in {0..9}
# do
#   acc=$(python inference.py --model cifar10 --shards 10 --slices 1 --dataset datasets/cifar10/datasetfile --batch_size 64 --dropout_rate 0.2 --container "cifar10" --unlearn_shards ${i} | tail -n 1)
#   cat containers/cifar10/times/shard-*.time > "containers/cifar10/times/times.tmp"
#   time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
#   echo "10,${i},${acc},${time}" >> cifar10-general-report.csv 
# done

### Unlearning n shards at a time.

# forgot_shards=($(shuf -i 0-9 -n 4))
# acc=$(python inference.py --model cifar10 --shards 10 --slices 1 --dataset datasets/cifar10/datasetfile --batch_size 64 --dropout_rate 0.2 --container "cifar10" --unlearn_shards $forgot_shards | tail -n 1)
# cat containers/cifar10/times/shard-*.time > containers/cifar10/times/times.tmp
# time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
# echo "10,\"[$(IFS=,; echo "${forgot_shards[*]}")]\",${acc},${time}" >> cifar10-general-report.csv