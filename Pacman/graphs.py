from save_ragged import *
import numpy as np
import os
from params import *
from functions import *
import matplotlib.pyplot as plt
import sys

experiment_number = 7

params_as_arguments = False
if params_as_arguments:
    observation_radius =  sys.argv[1]
    critical_threshold =  sys.argv[2]
    choice_of_C = sys.argv[3]
    experiment_number = sys.argv[4]

##################
# Initialization #
##################

##Initialize Counters/Trackers
number_of_fails=0
failed_path=[]
fail_history=[]
success_history=[]
number_of_successes=0
path_history=[]
final_states=[]
successful_episodes = []

##Q function. A 6-dim matrix (action,X,Y,X,Y,automaton_state) (X,Y,X,Y) = (pacmanstate,ghoststate)
Q=np.zeros((total_number_of_actions,X_limit,Y_limit,X_limit,Y_limit,5))

#State Counts
state_count=np.zeros((X_limit,Y_limit,X_limit,Y_limit))
state_count[pac_start[0],pac_start[1],g_start[0],g_start[1]] = 1

#Instead of state_action_state_counts, we have prior and posterior (where
#posterior-prior = counts. We just continuously update posterior by adding
#counts, and can subtract the prior at any point to get the counts.

#Defining the Prior
if prior_choice == "Uninformative Prior":
    #CAREFUL ABOUT DTYPES
    #THIS IS INTRACTABLE?! MAY NEED SPARCE MATRICES.
    state_action_direction_prior=np.zeros((X_limit,Y_limit,X_limit,Y_limit,total_number_of_actions,total_number_of_directions),dtype=np.int16)
    for (pacx,pacy,gx,gy,action_indx,direction),_ in np.ndenumerate(state_action_direction_prior):
        if not (wall_mat[pacx,pacy] or wall_mat[gx,gy]):
            next_state,hit_wall = take_direction_wall(np.array([pacx,pacy,gx,gy,0]),direction)
            if hit_wall == False:
                action_indx_as_dir = action_to_implied_direction(action_indx)
                known_next_pac_state,_ = move_agent_direction(np.array([pacx,pacy]),action_indx_as_dir)
                if (known_next_pac_state == next_state[0:2]).all():
                    state_action_direction_prior[pacx,pacy,gx,gy,action_indx,direction]=1

    state_action_direction_posterior = state_action_direction_prior

if prior_choice == "Full Prior":
    #CAREFUL ABOUT DTYPES
    #THIS IS INTRACTABLE?! MAY NEED SPARCE MATRICES.
    state_action_direction_prior=np.zeros((X_limit,Y_limit,X_limit,Y_limit,total_number_of_actions,total_number_of_directions),dtype=np.int16)
    for (pacx,pacy,gx,gy,action_indx,direction),_ in np.ndenumerate(state_action_direction_prior):
        if not (wall_mat[pacx,pacy] or wall_mat[gx,gy]):
            next_state,hit_wall = take_direction_wall(np.array([pacx,pacy,gx,gy,0]),direction)
            if hit_wall == False:
                action_indx_as_dir = action_to_implied_direction(action_indx)
                known_next_pac_state,_ = move_agent_direction(np.array([pacx,pacy]),action_indx_as_dir)
                g_next, g_dir = chase_pacman(known_next_pac_state,np.array([gx,gy]))
                if (known_next_pac_state == next_state[0:2]).all() and (g_next == next_state[2:4]).all():
                    state_action_direction_prior[pacx,pacy,gx,gy,action_indx,direction]=1

    state_action_direction_posterior = state_action_direction_prior

#Defining the Prior
if prior_choice == "Completely Uninformative Prior":
    #CAREFUL ABOUT DTYPES
    #THIS IS INTRACTABLE?! MAY NEED SPARCE MATRICES.
    state_action_direction_prior=np.zeros((X_limit,Y_limit,X_limit,Y_limit,total_number_of_actions,total_number_of_directions),dtype=np.int16)
    for (pacx,pacy,gx,gy,action_indx,direction),_ in np.ndenumerate(state_action_direction_prior):
        if not (wall_mat[pacx,pacy] or wall_mat[gx,gy]):
            next_state,hit_wall = take_direction_wall(np.array([pacx,pacy,gx,gy,0]),direction)
            if hit_wall == False:
                state_action_direction_prior[pacx,pacy,gx,gy,action_indx,direction]=1

    state_action_direction_posterior = state_action_direction_prior

#State_action_prior: similar to the above, based on the prior given above
state_action_prior=np.sum(state_action_direction_prior,(5))
state_action_posterior = state_action_prior


#Expected transition probabilities
#Note this gives many nans, for example on places where one is a wall.
ps = np.divide(state_action_direction_prior, np.expand_dims(state_action_prior,(5)))
#Maybe unnecessary (hopefully!)
ps = np.nan_to_num(ps, nan=0)

cov=np.zeros((X_limit,Y_limit,X_limit,Y_limit,total_number_of_actions,total_number_of_directions,total_number_of_directions)) #Can we reduce dtype?
for (pacx,pacy,gx,gy,a,diri,dirj),_ in np.ndenumerate(cov):
    if not (wall_mat[pacx,pacy] or wall_mat[gx,gy]):
        cov[pacx,pacy,gx,gy,a,diri,dirj] = ((diri == dirj)*ps[pacx,pacy,gx,gy,a,diri] \
            - ps[pacx,pacy,gx,gy,a,diri]*ps[pacx,pacy,gx,gy,a,dirj]) / (state_action_posterior[pacx,pacy,gx,gy,a] + 1)

#Defining the parameter C as a function of state_count
if choice_of_C == "ThesisDecreasing":
    def C_function(cur_state_count):
        return 0.7*max((24 - cur_state_count)/25,0) + 0.3/(1 + cur_state_count)


#LoadFilePath Initializations
basepath = os.getcwd()
results_path = os.path.join(basepath,'Results')
experiment_name = "pacman_depth{depth}_Pmax{Pmax}_C{C}_experiment_number{number}".format( \
    depth = observation_radius,Pmax = critical_threshold,C = choice_of_C,number = experiment_number)


##################
# Plotting       #
##################

#Load Data
fail_history = np.load(os.path.join(results_path,experiment_name + "_failHistory.npy"))
success_history = np.load(os.path.join(results_path,experiment_name + "_successHistory.npy"))
successful_episodes = np.load(os.path.join(results_path,experiment_name + "_successfulepisodes.npy"))
Q = np.load(os.path.join(results_path,experiment_name + "_Q.npy"))
pathHistory = load_stacked_arrays(os.path.join(results_path,experiment_name + "_pathHistory.npz"), axis=0)
pathLengths = [len(path) for path in pathHistory]
final_states = np.load(os.path.join(results_path,experiment_name + "_finalstates.npy"))


relevant_Q = Q[:,:,:,0]
Q_maxes = np.max(relevant_Q,axis=0)

#Reconstruct which episodes were fails:
failures = 0
fail_episodes = np.zeros(len(fail_history))
for i,fails in enumerate(fail_history):
    if fails > failures:
        failures += 1
        fail_episodes[i] = 1

successLengths = pathLengths * successful_episodes + max_steps * (1 - successful_episodes)
average_over = 50
running_average_successLengths = [ np.average(successLengths[i:i+average_over]) for i in np.arange(len(successLengths)) - average_over]

plots_path = os.path.join(basepath,'Plots')

state_count_product_mdp = np.zeros((X_limit,Y_limit,X_limit,Y_limit,5))
for path in pathHistory:
    for state in path:
        state_count_product_mdp[state[0],state[1],state[2],state[3],state[4]] += 1

state_count_physical = np.sum(state_count_product_mdp, (2,3,4))


plots_path = os.path.join(basepath,'Plots')

plt.figure()
plt.title("Risk Horizon m={riskHorizon}".format(riskHorizon=observation_radius))
#plt.title("Q-Learning Only")
plt.ylabel("steps taken to win")
plt.xlabel("episode number")
plt.plot(successLengths, label = 'data')
# plt.plot(22*np.ones((len(successLengths))),'--', label = 'optimal', )
plt.plot(running_average_successLengths, label = '{n}-running average'.format(n=average_over))
plt.legend()
#plt.setp(plot1[1],)
plt.savefig(os.path.join(plots_path,experiment_name + "_successLengths.png"))


plt.figure()
plt.plot(fail_history)
plt.savefig(os.path.join(plots_path,experiment_name + "_failHistory.png"))

plt.figure()
plt.matshow(state_count_physical)
plt.grid()
plt.savefig(os.path.join(plots_path,experiment_name + "_stateCountPhysical.png.png"))

# plt.figure()
# plt.matshow(Q_maxes)
# plt.savefig(os.path.join(plots_path,experiment_name + "_Qfunction.png.png"))

plt.figure()
plt.plot(successful_episodes)
plt.savefig(os.path.join(plots_path,experiment_name + "__successfulepisodes.png"))

print("Finished Running graphs.py")