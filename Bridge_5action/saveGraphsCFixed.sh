#!/bin/bash

#critical_threshold=0.33
#choice_of_C="Fixed0.10"

for choice_of_C in "Fixed0.01" "Fixed0.10" "Fixed0.30" "Fixed0.70"
do
    for critical_threshold in 0.01 0.033 0.1 0.33
    do
        for experiment_num in {1..5}
        do
            conda run -n minimal_ds python graphs.py $critical_threshold $choice_of_C $experiment_num
            printf "Plotted $experiment_num of 5 runs\n"
        done
    done
done