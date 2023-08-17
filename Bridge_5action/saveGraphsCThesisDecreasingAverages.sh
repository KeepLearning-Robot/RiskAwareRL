#!/bin/bash

#critical_threshold=0.33
choice_of_C="ThesisDecreasing"
experiment_num_max=5

for critical_threshold in 0.56 0.33 0.18 0.1 0.033 0.01
do
    conda run -n minimal_ds python graphAverages.py $critical_threshold $choice_of_C 1 $experiment_num_max
    printf "Plotted $experiment_num of 5 runs\n"
done
