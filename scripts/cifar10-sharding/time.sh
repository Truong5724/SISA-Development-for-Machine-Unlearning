set -eou pipefail
IFS=$'\n\t'

if [[ ! -f cifar10-general-report.csv ]]; then
    echo "nb_shards, retraining_time" > cifar10-general-report.csv
fi

time=$(python time_stats.py --container "cifar10" | awk -F ',' '{print $1}')
echo "10,${time}" >> cifar10-general-report.csv