#!/bin/bash

choice_of_C="ThesisDecreasing"
critical_threshold=0.01
prior_choice="Uninformative Prior"


for experiment_num_max in 10
do
    conda run -n minimal_ds python graphAverages.py $critical_threshold $choice_of_C 1 $experiment_num_max $prior_choice
    printf "Plotted averages of 1 to $experiment_num_max \n"
done
