from save_ragged import *
import numpy as np
import os
from params import *
from functions import *
import matplotlib.pyplot as plt
import sys

params_as_arguments = True
if params_as_arguments:
    critical_threshold =  sys.argv[1]
    choice_of_C = sys.argv[2]
    experiment_number = sys.argv[3]


##################
# Initialization #
##################

##Defining the area
X_movable_limit = 20
Y_movable_limit = 20
X=np.linspace(1,X_movable_limit,X_movable_limit)
Y=np.linspace(1,Y_movable_limit,Y_movable_limit)
[gridx,gridy]=np.meshgrid(X,Y)

##State Labels
u=np.zeros([20,20])
u[0:7,0:20]=2
u[8:12,0:8]=1
u[8:12,11:20]=1

##Initialize Counters/Trackers
it_run_num=1
number_of_fails=0
failed_path=[]
fail_history=[]
success_history=[]
number_of_successes=0
path_history=[]
successful_runs = np.zeros(number_of_runs)

##Q function. A 4-dim matrix (action,state(X, Y),automaton_state)
total_number_of_actions=5
Q=np.zeros((total_number_of_actions,X_movable_limit,Y_movable_limit,3)) #Is there a point having the automaton_state here? Maybe just for generality?

#State Counts
state_count=np.zeros((X_movable_limit,Y_movable_limit))
state_count[19,0] = 1

#Instead of state_action_state_counts, we have prior and posterior (where
#posterior-prior = counts. We just continuously update posterior by adding
#counts, and can subtract the prior at any point to get the counts.

#Defining the Prior
if prior_choice == "Uninformative Prior":
    state_action_direction_prior=np.zeros((20,20,5,5))
    for action_indx in np.arange(5):
        for x in np.arange(0,X_movable_limit):
            for y in np.arange(0,Y_movable_limit):
                for direction in np.arange(0,5):
                    poss_next_state,out_of_bounds = take_action_m_boundary([x,y,1],direction,True,u)
                    if out_of_bounds == False:
                        state_action_direction_prior[x,y,action_indx,direction]=1
    state_action_direction_posterior = state_action_direction_prior

if prior_choice == "Medium Informative Prior":
    state_action_direction_prior=np.zeros((20,20,5,5))
    for action_indx in np.arange(5):
        for x in np.arange(0,X_movable_limit):
            for y in np.arange(0,Y_movable_limit):
                for direction in np.arange(0,5):
                    poss_next_state,out_of_bounds = take_action_m_boundary([x,y,1],direction,True,u)
                    if out_of_bounds == False:
                        if action_indx == direction:
                            state_action_direction_prior[x,y,action_indx,direction]=12
                        else:
                            state_action_direction_prior[x,y,action_indx,direction]=1
    state_action_direction_posterior = state_action_direction_prior

if prior_choice == "High Informative Prior":
    state_action_direction_prior=np.zeros((20,20,5,5))
    for action_indx in np.arange(5):
        for x in np.arange(0,X_movable_limit):
            for y in np.arange(0,Y_movable_limit):
                for direction in np.arange(0,5):
                    poss_next_state,out_of_bounds = take_action_m_boundary([x,y,1],direction,True,u)
                    if out_of_bounds == False:
                        if action_indx == direction:
                            state_action_direction_prior[x,y,action_indx,direction]=96
                        else:
                            state_action_direction_prior[x,y,action_indx,direction]=1
    state_action_direction_posterior = state_action_direction_prior

#State_action_prior: similar to the above, based on the prior given above
state_action_prior=np.sum(state_action_direction_prior,3)
state_action_posterior = state_action_prior

#Expected transition probabilities
ps = np.divide(state_action_direction_prior, np.expand_dims(state_action_prior,3))

#Covariance Matrix: cov(q1,q2,action,:,:) is the 5*5 covariance matrix for
#the dirichlet for taking action at state (q1,q2)
cov = np.zeros((20,20,5,5,5))
for q1 in np.arange(0,20):
    for q2 in np.arange(0,20):
        for a in np.arange(0,5):
            for direction_i in np.arange(0,5):
                for direction_j in np.arange(0,5):
                    cov[q1,q2,a,direction_i,direction_j] = ((direction_i == direction_j)*ps[q1,q2,a,direction_i] - ps[q1,q2,a,direction_i]*ps[q1,q2,a,direction_j]) \
                       / (state_action_posterior[q1,q2,a] + 1)

#Defining the parameter C as a function of state_count
if choice_of_C == "ThesisDecreasing":
    def C_function(cur_state_count):
        return 0.7*max((24 - cur_state_count)/25,0) + 0.3/(1 + cur_state_count)

if choice_of_C[0:5] == "Fixed":
    c = float(choice_of_C[5:9])
    def C_function(cur_state_cout):
        return c

if choice_of_C[0:5] == "Slope":
    num = int(choice_of_C[5:7])
    def C_function(cur_state_count):
        return 0.7*max(((num-1) - cur_state_count)/num,0) + 0.3/(1 + cur_state_count)

#LoadFilePath Initializations
basepath = os.getcwd()
results_path = os.path.join(basepath,'Results')
experiment_name = "bridge_Pmax{Pmax}_C{C}_Prior{prior}_experiment{number}".format( \
    Pmax = critical_threshold,prior = prior_choice[0:4],C = choice_of_C,number=experiment_number)

#Load Data
fail_history = np.load(os.path.join(results_path,experiment_name + "_failHistory.npy"))
success_history = np.load(os.path.join(results_path,experiment_name + "_successHistory.npy"))
successful_runs = np.load(os.path.join(results_path,experiment_name + "_successfulRuns.npy"))
Q = np.load(os.path.join(results_path,experiment_name + "_Q.npy"))
pathHistory = load_stacked_arrays(os.path.join(results_path,experiment_name + "_pathHistory.npz"), axis=0)
pathLengths = [len(path) for path in pathHistory]

relevant_Q = Q[:,:,:,0]
Q_maxes = np.max(relevant_Q,axis=0)

#Reconstruct which runs were fails:
failures = 0
fail_runs = np.zeros(len(fail_history))
for i,fails in enumerate(fail_history):
    if fails > failures:
        failures += 1
        fail_runs[i] = 1

successLengths = pathLengths * successful_runs

plots_path = os.path.join(basepath,'Plots')

plt.figure()
plt.plot(successLengths)
plt.savefig(os.path.join(plots_path,experiment_name + "_successLengths.png"))

plt.figure()
plt.plot(fail_history)
plt.savefig(os.path.join(plots_path,experiment_name + "_failHistory.png"))

plt.figure()
plt.matshow(Q_maxes)
plt.savefig(os.path.join(plots_path,experiment_name + "_Qfunction.png.png"))
