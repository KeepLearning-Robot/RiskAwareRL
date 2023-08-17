import numpy as np
import sys
import os
from params import *
from functions import *
from save_ragged import *
#import matplotlib.pyplot as plt

#Allow user commandline to set experiment_number
experiment_number =  sys.argv[1]

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

#SaveFilePath Initializations
basepath = os.getcwd()
results_path = os.path.join(basepath,'Results')
experiment_name = "pacman_depth{depth}_Pmax{Pmax}_C{C}_experiment_number{number}".format( \
    depth = observation_radius,Pmax = critical_threshold,C = choice_of_C,number = experiment_number)

##################
# Train Agent    #
##################

#Train Agent
ep_num = 0
while ep_num < number_of_episodes and ((number_of_successes/(ep_num+1) < success_rate_objective) or ep_num < 20):
    successful_episodes.append(0)
    ep_num += 1
    print('Episode Number: ' + str(ep_num))
    current_state = np.array([pac_start[0],pac_start[1],g_start[0],g_start[1],0]) #In the product MDP
    #current_state = np.array([1,3,1,4,0]) #TESTING
    current_path = np.expand_dims(current_state,0) #Add a dimension so we can keep a list of states
    step_num = 1
    while (current_state[-1]!=3) and (step_num <= max_steps):
        print('step Number: {number}'.format(number=step_num))
        step_num = step_num + 1
        if safe_padding:
            depth = observation_radius # Should be <= observation_radius
            old_neighbours=np.expand_dims(current_state,0) #Add a dimension so we can keep a list of states
            neighbours=np.array([]).astype(np.int16).reshape(0,5) 
            #Expand neighbours one at a time until covering the
            #Obervation_radius: make sure it is unique.
            #NOTE: Neighbours is in the product MDP
            

            for i in np.arange(observation_radius):
                for current_exploring in np.arange(np.size(old_neighbours,0)):
                    for direction in np.arange(total_number_of_directions):
                        neighbours= np.vstack([neighbours,take_direction(old_neighbours[current_exploring],direction)])
                    neighbours=np.unique(neighbours,axis=0)
                old_neighbours=np.vstack([old_neighbours,neighbours])
                old_neighbours=np.unique(old_neighbours,axis=0)

            #RISK AND VARIANCE CALCULATION!
            # Dummy
            #U,variance_U = np.zeros(total_number_of_actions),np.zeros(total_number_of_actions)         
            U,variance_U=local_U_update(current_state,neighbours,depth,ps,cov)
            

            #Defining the parameter C for risk-averseness based on state_count
            C = C_function(state_count[current_state[0],current_state[1],current_state[2],current_state[3]])

            #Perform Cantelli Bound calculation
            P_a = U + np.sqrt((variance_U*C)/(1-C))

            #(weird formatting because np.nonzero returns index tuples)
            (acceptable_actions,)=np.nonzero(P_a<critical_threshold)
            #If nothing is acceptable, then just choose minimum U.
            if acceptable_actions.size == 0:
                acceptable_actions = np.nonzero(U == np.amin(U))


        available_Qs = np.zeros(total_number_of_actions)
        if safe_padding == 1:
            for each_action in acceptable_actions:
                available_Qs[each_action]=Q[each_action,current_state[0],current_state[1],current_state[2],current_state[3],current_state[4]]
            for each_action in np.setdiff1d(np.arange(total_number_of_actions),acceptable_actions,assume_unique=True):
                available_Qs[each_action]=Q[each_action,current_state[0],current_state[1],current_state[2],current_state[3],current_state[4]] - 300
        else:
            for each_action in np.arange(total_number_of_actions):
                available_Qs[each_action]=Q[each_action,current_state[0],current_state[1],current_state[2],current_state[3],current_state[4]]

        #Boltzmann rational
        expo=np.exp(available_Qs/temp)
        probabs=expo/sum(expo) 

        #Select an action
        actions=np.arange(total_number_of_actions)
        selected_action = np.random.choice(actions, p=probabs)

        #print('neigh: {neighbours}'.format(neighbours=neighbours))
        print('State: {state}'.format(state=current_state))
        #print('Dirs : Right, Up   , Down , Left')
        #if safe_padding:
        #    print('VarU : {var}'.format(var=variance_U))
        #    print('U    : {U}'.format(U=U))
        #print('probs: {probs}'.format(probs=probabs))
        #print('actio: {action}'.format(action=selected_action))

        #Take that action
        next_state,direction_taken=take_action_direction(current_state,selected_action,False)
        
        #print('Stat2: {state}'.format(state=next_state))
        #print('dirTaken:{d}'.format(d=direction_taken))
        #print('1post: {p}'.format(p=state_action_direction_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,:]))

        #UPDATES
        #Update posterior by adding counts
        state_action_direction_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,direction_taken]= \
            state_action_direction_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,direction_taken] + 1
        
        #print('2post: {p}'.format(p=state_action_direction_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,:]))

        state_action_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action] = \
            state_action_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action] + 1
        #Update Means
        ps[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,:] = \
            state_action_direction_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,:] / state_action_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action]
        #Update Covariance Matrix
        for direction_i in np.arange(total_number_of_directions):
            for direction_j in np.arange(total_number_of_directions):
                cov[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,direction_i,direction_j] = ((direction_i == direction_j)*ps[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,direction_i] - ps[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,direction_i]*ps[current_state[0],current_state[1],current_state[2],current_state[3],selected_action,direction_j]) \
                    / (state_action_posterior[current_state[0],current_state[1],current_state[2],current_state[3],selected_action] + 1)
        
        #Update Q function
        current_Qs=Q[:,next_state[0],next_state[1],next_state[2],next_state[3],next_state[4]]
        
        Q[selected_action,current_state[0],current_state[1],current_state[2],current_state[3],current_state[4]] = (1-alpha)* \
            Q[selected_action,current_state[0],current_state[1],current_state[2],current_state[3],current_state[4]] + alpha * \
            (Q_r(current_state[4],next_state[4]) + discount_factor_Q*np.amax(current_Qs)) 

        #Update state counts to reflect moving to next state
        state_count[next_state[0],next_state[1],next_state[2],next_state[3]] = state_count[next_state[0],next_state[1],next_state[2],next_state[3]] + 1
        current_path=np.vstack([current_path,next_state])

        #DISPLAY (AND TERMINATE IF FAILED)
        if next_state[4] == 4:
            number_of_fails = number_of_fails + 1
            print('-------fail-------')
            #print('Current State: ' + str(current_state))
            #print('Next State: ' + str(next_state))
            #print('Neighbours: ' + str(neighbours))
            #print('U: ' + str(U))
            #print('Selected Action:' + str(selected_action))
            break
        elif next_state[4] == 3:
            print('+++++++success+++++++')
            number_of_successes = number_of_successes + 1
            #Keep a record of which episodes were successful.
            successful_episodes[ep_num - 1] = 1
        
        #Update current state
        current_state = next_state.copy()

            
    #Add episode to paths.        
    path_history.append(current_path)

    #Cumulative total of number of successes.
    fail_history.append(number_of_fails)
    success_history.append(number_of_successes)
    final_states.append(next_state)

##################
# Print and Save #
##################

#Prints
print('Number of Successes: ' + str(number_of_successes))
print('Number of Failures: ' + str(number_of_fails))    

#Save Results!           
np.save(os.path.join(results_path,experiment_name + "_failHistory"), fail_history)
np.save(os.path.join(results_path,experiment_name + "_successHistory"), success_history)
np.save(os.path.join(results_path,experiment_name + "_successfulepisodes"), successful_episodes)
np.save(os.path.join(results_path,experiment_name + "_Q"), Q)
save_stacked_array(os.path.join(results_path,experiment_name + "_pathHistory"), path_history, axis=0)
np.save(os.path.join(results_path,experiment_name + "_finalStates"), final_states)

#Plot
#plt.plot(fail_history)
#plt.show()
