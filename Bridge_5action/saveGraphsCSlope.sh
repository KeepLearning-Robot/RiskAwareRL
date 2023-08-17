#!/bin/bash

critical_threshold=0.33
#choice_of_C="ThesisDecreasing"


for choice_of_C in "Slope10" "Slope20" "Slope30" "Slope40"
do
    for experiment_num in {1..5}
    do
        conda run -n minimal_ds python graphs.py $critical_threshold $choice_of_C $experiment_num
        printf "Plotted $experiment_num of 5 runs\n"
    done
done
