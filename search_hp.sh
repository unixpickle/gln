#!/bin/bash

log_dir=hp_logs
if ! [ -d "$log_dir" ]; then
    mkdir "$log_dir"
fi
for lr in 0.1 0.05 0.01 0.001; do
    for half_spaces in 1 2 3 4; do
        out_file="$log_dir/lr${lr}_hs${half_spaces}.txt"
        echo "lr: $lr  half_spaces: $half_spaces"
        python train_mnist.py \
            --half-spaces $half_spaces \
            --lr $lr \
            --deterministic \
            >"$out_file"
        echo '*' $(cat "$out_file" | grep "test accuracy")
    done
done
