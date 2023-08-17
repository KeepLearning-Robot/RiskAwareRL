#!/bin/bash

for experiment_num in {1..1}
do
    conda run -n minimal_ds python main.py $experiment_num
    printf "Finished $experiment_num of 1 experiments\n"
done
