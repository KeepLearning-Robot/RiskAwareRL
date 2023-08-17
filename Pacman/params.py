import numpy as np
#Experiment Parameters
safe_padding = True 
number_of_episodes = 50
max_steps = 400
experiment_number = 1
observation_radius=2
success_rate_objective = 0.3 #Make this lower than 1 to terminate early if we get successful enough.

##Hyper Parameters
discount_factor_Q=0.9
discount_factor_U=1
omega=0.79
alpha=0.9
softmax = 1
temp=.005

#Choice of Prior
prior_choice = "Uninformative Prior"
#Choice of Critical Threshold (Pmax)
critical_threshold=0.01
#Choice of parameter C for risk-averseness (function of state_count). See main.py initialization section for exact meanings.
#choice_of_C = "Slope10"
choice_of_C = "ThesisDecreasing"    