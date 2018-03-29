#!/bin/bash

# run.sh

mkdir -p _results/cub

for i in $(seq 20); do
    python cub.py --seed 1111$i > _results/cub/results.$i.json
done
