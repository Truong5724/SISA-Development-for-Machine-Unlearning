#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

python init.py --shards 10 --container "cifar10" 

