import numpy as np

##Experiment Parameters
safe_padding = True # make it True to activate padding
number_of_runs = 500
max_iterations = 400
experiment_number = 1

##Hyper Parameters
discount_factor_Q=0.9
discount_factor_U=1
alpha=0.85
max_horizon=4      
m_3=30
m_2=30
m_1=max_horizon-0.01/np.exp(-1/m_2)
observation_radius=2
softmax = 1
temp=.005

#Choice of Prior
prior_choice = "Uninformative Prior"
#Choice of Critical Threshold (Pmax)
critical_threshold=0.1
#Choice of parameter C for risk-averseness (function of state_count)
#choice_of_C = "Slope10"
choice_of_C = "ThesisDecreasing"

