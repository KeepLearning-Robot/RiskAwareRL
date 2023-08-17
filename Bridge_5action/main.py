#import matplotlib.pyplot as plt
from save_ragged import *
import sys
import os
import numpy as np
from params import *
from functions import *

#Allow user commandline to set experiment_number
experiment_number =  sys.argv[1]

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
    
#SaveFilePath Initializations
basepath = os.getcwd()
results_path = os.path.join(basepath,'Results')
experiment_name = "bridge_Pmax{Pmax}_C{C}_Prior{prior}_experiment{number}".format( \
    Pmax = critical_threshold,prior = prior_choice[0:4],C = choice_of_C,number=experiment_number)

##################
# Train Agent    #
##################

#Train Agent
for run_num in np.arange(1,number_of_runs + 1):
    current_state = np.array([19,0,0]) #In the product MDP
    current_path = np.expand_dims(current_state,0) #Add a dimension so we can keep a list of states
    print('Run Number: ' + str(run_num))
    it_num = 1
    while (current_state[-1]!=2) and (it_num <= max_iterations):
        #SOME DEBUGGING PRINTS
        #print('Iteration Number: ' + str(it_num))
        #print('Current State: ' + str(current_state))
        it_num = it_num + 1
        it_run_num = it_run_num+1

        if safe_padding:
            depth = 2 #Must be <= observation radius
            old_neighbours=np.expand_dims(current_state,0) #Add a dimension so we can keep a list of states
            neighbours=np.array([]).astype(np.int8).reshape(0,3) #CAREFUL IF THE SIZE OF THE AREA IS LARGER THAN 128
            #Expand neighbours one at a time until covering the
            #Obervation_radius: make sure it is unique.
            #NOTE: Neighbours is in the product MDP
            for i in np.arange(observation_radius):
                for current_exploring in np.arange(np.size(old_neighbours,0)):
                    for action in np.arange(5):
                        neighbours= np.vstack([neighbours,take_action_m(old_neighbours[current_exploring],action,True,u)])
                    neighbours=np.unique(neighbours,axis=0)
                old_neighbours=np.vstack([old_neighbours,neighbours])
                old_neighbours=np.unique(old_neighbours,axis=0)
            
            #RISK AND VARIANCE CALCULATION!            
            U,variance_U=local_U_update(current_state,neighbours,depth,u,ps,cov)

            #Defining the parameter C for risk-averseness based on state_count
            C = C_function(state_count[current_state[0],current_state[1]])

            #Perform Cantelli Bound calculation
            P_a = U + np.sqrt((variance_U*C)/(1-C))

            #(weird formatting because np.nonzero returns index tuples)
            (acceptable_actions,)=np.nonzero(P_a<critical_threshold)
            #If nothing is acceptable, then just choose minimum U.
            if acceptable_actions.size == 0:
                acceptable_actions = np.nonzero(U == np.amin(U))
            
        available_Qs = np.zeros(5)
        if safe_padding == 1:
            for each_action in acceptable_actions:
                available_Qs[each_action]=Q[each_action,current_state[0],current_state[1],current_state[2]]
            for each_action in np.setdiff1d(np.arange(5),acceptable_actions,assume_unique=True):
                available_Qs[each_action]=Q[each_action,current_state[0],current_state[1],current_state[2]] - 300
        else:
            for each_action in np.arange(5):
                available_Qs[each_action]=Q[each_action,current_state[0],current_state[1],current_state[2]]

        #Boltzmann rational
        expo=np.exp(available_Qs/temp)
        probabs=expo/sum(expo)  

        #Select an action
        actions=np.arange(5)
        selected_action = np.random.choice(actions, p=probabs)

        #Take that action
        next_state,direction_taken=take_action_m_direction(current_state,selected_action,False,u)

        #UPDATES
        #Update posterior by adding counts
        state_action_direction_posterior[current_state[0],current_state[1],selected_action,direction_taken]= \
            state_action_direction_posterior[current_state[0],current_state[1],selected_action,direction_taken] + 1
        
        state_action_posterior[current_state[0],current_state[1],selected_action] = \
            state_action_posterior[current_state[0],current_state[1],selected_action] + 1
        #Update Means
        ps[current_state[0],current_state[1],selected_action,:] = \
            state_action_direction_posterior[current_state[0],current_state[1],selected_action,:] / state_action_posterior[current_state[0],current_state[1],selected_action]
        #Update Covariance Matrix
        for direction_i in np.arange(5):
            for direction_j in np.arange(5):
                cov[current_state[0],current_state[1],selected_action,direction_i,direction_j] = ((direction_i == direction_j)*ps[current_state[0],current_state[1],selected_action,direction_i] - ps[current_state[0],current_state[1],selected_action,direction_i]*ps[current_state[0],current_state[1],selected_action,direction_j]) \
                    / (state_action_posterior[current_state[0],current_state[1],selected_action] + 1)
        
        #Update Q function
        current_Qs=Q[:,next_state[0],next_state[1],next_state[2]]
        
        Q[selected_action,current_state[0],current_state[1],current_state[2]] = (1-alpha)* \
            Q[selected_action,current_state[0],current_state[1],current_state[2]] + alpha * \
            (Q_r(next_state) + discount_factor_Q*np.amax(current_Qs)) 

        #Update state counts to reflect moving to next state
        state_count[next_state[0],next_state[1]] = state_count[next_state[0],next_state[1]] + 1
        current_path=np.vstack([current_path,next_state])

        #DISPLAY (AND TERMINATE IF FAILED)
        if next_state[2] == 1:
            number_of_fails = number_of_fails + 1
            #print('-------fail-------')
            #print('Current State: ' + str(current_state))
            #print('Next State: ' + str(next_state))
            #print('Neighbours: ' + str(neighbours))
            #print('U: ' + str(U))
            #print('Selected Action:' + str(selected_action))
            break
        elif next_state[2] == 2:
            #print('+++++++success+++++++')
            number_of_successes = number_of_successes + 1
            #Keep a record of which runs were successful.
            successful_runs[run_num - 1] = 1
        
        #Update current state
        current_state = next_state.copy()



            
    #Add run to paths.        
    path_history.append(current_path)

    #Cumulative total of number of successes.
    fail_history.append(number_of_fails)
    success_history.append(number_of_successes)

#Prints
print('Number of Successes: ' + str(number_of_successes))
print('Number of Failures: ' + str(number_of_fails))    

#Save Results!           
np.save(os.path.join(results_path,experiment_name + "_failHistory"), fail_history)
np.save(os.path.join(results_path,experiment_name + "_successHistory"), success_history)
np.save(os.path.join(results_path,experiment_name + "_successfulRuns"), successful_runs)
np.save(os.path.join(results_path,experiment_name + "_Q"), Q)
save_stacked_array(os.path.join(results_path,experiment_name + "_pathHistory"), path_history, axis=0)

#Plot
#plt.plot(fail_history)
#plt.show()
