#!/bin/bash

for experiment_num in {1..5}
do
    conda run -n minimal_ds python main.py $experiment_num
    printf "Finished $experiment_num of 5 runs\n"
done
