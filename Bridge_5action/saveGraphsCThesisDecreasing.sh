#!/bin/bash

#critical_threshold=0.33
choice_of_C="ThesisDecreasing"


for critical_threshold in 0.56 0.33 0.18 0.1 0.033 0.01
do
    for experiment_num in {1..5}
    do
        conda run -n minimal_ds python graphs.py $critical_threshold $choice_of_C $experiment_num
        printf "Plotted $experiment_num of 5 runs\n"
    done
done
