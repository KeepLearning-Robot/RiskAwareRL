import numpy as np

def automaton(u_val,last_automaton_state):
    #1 is unsafe and 2 is target
    #If we were on a neutral state
    if last_automaton_state == 0:
        next_automaton_state = u_val
    else: #We were already either unsafe or target
        next_automaton_state = last_automaton_state
    return next_automaton_state

def take_action_m(current_location,action_indx,is_det,u):
    next_location = current_location.copy()
    if (not is_det):
        if (np.random.uniform() > .95):
            action_indx = np.floor(5*np.random.uniform())
    if action_indx == 0: #Right
        next_location[1] = current_location[1]+1
    elif action_indx == 1: #Up
        next_location[0] = current_location[0]-1
    elif action_indx == 2: #Left
        next_location[1] = current_location[1]-1
    elif action_indx == 3: #Down
        next_location[0] = current_location[0]+1
    # Else if action is 4, Stay.
    if next_location[0] > 19:
       next_location[0] = 19
    if next_location[0] < 0:
       next_location[0] = 0
    if next_location[1] > 19:
       next_location[1] = 19
    if next_location[1] < 0:
       next_location[1] = 0
    next_location[2] = automaton(u[next_location[0],next_location[1]],current_location[2])
    return next_location

def take_action_m_boundary(current_location,action_indx,is_det,u):
    out_of_bounds = False
    next_location = current_location.copy()
    if (not is_det):
        if (np.random.uniform() > .95):
            action_indx = np.floor(5*np.random.uniform())
    if action_indx == 0: #Right
        next_location[1] = current_location[1]+1
    elif action_indx == 1: #Up
        next_location[0] = current_location[0]-1
    elif action_indx == 2: #Left
        next_location[1] = current_location[1]-1
    elif action_indx == 3: #Down
        next_location[0] = current_location[0]+1
    # Else if action is 4, Stay.
    if next_location[0] > 19:
       next_location[0] = 19
       out_of_bounds = True
    if next_location[0] < 0:
       next_location[0] = 0
       out_of_bounds = True
    if next_location[1] > 19:
       next_location[1] = 19
       out_of_bounds = True       
    if next_location[1] < 0:
       next_location[1] = 0
       out_of_bounds = True
    next_location[2] = automaton(u[next_location[0],next_location[1]],current_location[2])
    return next_location,out_of_bounds

#For vectorizing local_U_update_fixed_actions
#@profile
def take_actions_boundary_det(current_location,u):
    out_of_bounds = np.zeros(5)
    next_locations = np.tile(current_location, (5,1))

    action_to_movement = np.array([[0,1],[-1,0],[0,-1],[1,0],[0,0]])
    next_locations[:,0:2] += action_to_movement[[0,1,2,3,4]]
    
    #Couldnt figure out a good way around this loop
    for i in np.arange(5):
        if next_locations[i,0] > 19:
            next_locations[i,0] = 19
            out_of_bounds[i] = True
        if next_locations[i,0] < 0:
            next_locations[i,0] = 0
            out_of_bounds[i] = True
        if next_locations[i,1] > 19:
            next_locations[i,1] = 19
            out_of_bounds[i] = True       
        if next_locations[i,1] < 0:
            next_locations[i,1] = 0
            out_of_bounds[i] = True
        next_locations[i,2] = automaton(u[next_locations[i,0],next_locations[i,1]],current_location[2])
    return next_locations,out_of_bounds

def take_action_m_direction(current_location,action_indx,is_det,u):
    next_location = current_location.copy()
    if (not is_det):
        if (np.random.uniform() > .95):
            action_indx = np.int(5*np.random.uniform())
    if action_indx == 0: #Right
        next_location[1] = current_location[1]+1
    elif action_indx == 1: #Up
        next_location[0] = current_location[0]-1
    elif action_indx == 2: #Left
        next_location[1] = current_location[1]-1
    elif action_indx == 3: #Down
        next_location[0] = current_location[0]+1
    # Else if action is 4, Stay.
    direction_taken = action_indx
    if next_location[0] > 19:
       next_location[0] = 19
       direction_taken = 4
    if next_location[0] < 0:
       next_location[0] = 0
       direction_taken = 4
    if next_location[1] > 19:
       next_location[1] = 19
       direction_taken = 4
    if next_location[1] < 0:
       next_location[1] = 0
       direction_taken = 4
    next_location[2] = automaton(u[next_location[0],next_location[1]],current_location[2])
    return next_location,direction_taken

#Reward function
def Q_r(next_state):
    if next_state[2] == 2:
        return 1
    else:
        return 0

#Local risk calcuation with fixed actions
#@profile
def local_U_update_fixed_actions(current_state,neighbours,unsafe_states,depth,ps_delta,min_actions,out_of_bounds_vec,next_state_indx_vec,in_neighbours_vec,current_out_of_bounds,current_next_states_indx_vec,current_in_neighbours_vec,U_start):
    discount = 1
    U = U_start.copy()

    #First depth-1 steps at once, with actions given by min_actions
    for d in np.arange(depth-1):
        actions = min_actions[d,:]
        sums=discount*ps_delta[neighbours[:,0],neighbours[:,1],actions,:]*U[next_state_indx_vec[:,:]]*(out_of_bounds_vec[:,:] == 0)*in_neighbours_vec[:,:]                
        U = np.sum(sums, axis = 1)

        #Ensure U is 1 for unsafe states.
        U[unsafe_states] = 1
   
    #FINAL STEP (fixed all actions)        
    sums1=discount*ps_delta[current_state[0],current_state[1],:,:]*U[current_next_states_indx_vec[:]]*(current_out_of_bounds == 0)*current_in_neighbours_vec              
    
    U_delta = np.sum(sums1,1)
    return U_delta

#Variance on Expected Risk Calculation
#@profile
def risk_variance(current_state,neighbours,unsafe_states,num_neighbours,depth,ps,cov,min_actions,U_s_a,out_of_bounds_vec,next_state_indx_vec,in_neighbours_vec,current_out_of_bounds,current_next_states_indx_vec,current_in_neighbours_vec,U_start):
    #Covariance Matrix based on lexicographic ordering of state,action,state on
    #transition probabilities
    #We only represent the non-zero entries in the covariance matrix.
    #So cov(q1,q2,a,direction1,direction2) gives the covariance between the
    #transitions probability (q1,q2) (a)-> direction1(q1,q2) and (q1,q2) (a)-> direction2(q1,q2)
    #Maybe we should maintain and update this rather than recalculating at
    #every step.

    #We choose delta for numerical differentiation, standard detla = 0.001
    delta = 0.001

    #Just calculate the relevant ones (transition probs starting in
    #neighbouring states - these are the only ones used in risk
    variance_U = np.zeros(5) 
        
    #Now calculate variance
    #Do simultaneously for each first_action
    for neigh in np.arange(num_neighbours):
        for a in np.arange(5):
            #For storing gradients
            grad = np.zeros((5,5))
            for d in np.arange(5):
                #Calculate derivative wrt. p_neigh_a_d
                ps_delta = ps.copy()
                ps_delta[neighbours[neigh,0],neighbours[neigh,1],a,d] = \
                    ps_delta[neighbours[neigh,0],neighbours[neigh,1],a,d] + delta

                #Replicate local_U_update calculation but with actions
                #according to min_actions.
                
                U_delta = local_U_update_fixed_actions(current_state,neighbours,unsafe_states,depth,ps_delta,min_actions,out_of_bounds_vec,next_state_indx_vec,in_neighbours_vec,current_out_of_bounds,current_next_states_indx_vec,current_in_neighbours_vec,U_start)
                #Row? vector of gradients for risk after each first_action, wrt
                #p_neigh_a_d.
                #Each column contains the gradients wrt p_neigh_a_allds for the
                #risk after a particular action
                grad[d,:] = (U_delta - U_s_a)/delta
            
            
            #Add contribution to variance. @ is matrix multiplication
            for first_action in np.arange(5):
                variance_U[first_action] = variance_U[first_action] + np.transpose(grad[:,first_action]) @ cov[neighbours[neigh,0],neighbours[neigh,1],a,:,:] @ grad[:,first_action]
    return variance_U

#Local Risk Calculation
#@profile
def local_U_update(current_state,neighbours,depth,u,ps,cov):
    discount = 1
    num_neighbours = np.shape(neighbours)[0]

    #Pre-Calculate as Much as We Can.
    U_start = np.zeros(num_neighbours)
    unsafe_states = neighbours[:,2] == 1
    U_start[unsafe_states] = 1
    
    #Pre-calculate next_state, next_state_indx, out_of_bounds and in_neighbours
    #for each neigh and direction
    next_state_vec = np.zeros((num_neighbours,5,3),dtype=np.int8)
    out_of_bounds_vec = np.zeros((num_neighbours,5),np.int8)
    next_state_indx_vec = np.zeros((num_neighbours,5),np.int8) #Defaults to 0 if not in neighbours. Won't influence calculation
    in_neighbours_vec = np.ones((num_neighbours,5),np.int8)
    for neigh in np.arange(num_neighbours):
        for direction in np.arange(5):
            next_state,out_of_bounds = take_action_m_boundary(neighbours[neigh],direction,True,u)
            next_state_vec[neigh,direction] = next_state
            out_of_bounds_vec[neigh,direction] = out_of_bounds
            
            (next_state_indx_array,) = np.where(np.all(neighbours == next_state, axis = 1))

            if len(next_state_indx_array) == 0:
                in_neighbours_vec[neigh,direction] = 0
                #next_state_indx_vec defaults to 0
            else:
                #in_neighbours defaults to 1
                next_state_indx_vec[neigh,direction] = next_state_indx_array[0]

    #Also pre-calculate next_states, next_states_indx, out_of_bounds and in_neighbours
    #For the final step, with CURRENT STATE and All Directions
    current_next_states,current_out_of_bounds = take_actions_boundary_det(current_state,u) #Note that out_of_bounds here is 1 dimensional, but is used in the multiplication as 2 dimensional, because it is the same over actions.
    current_next_states_indx_vec = np.zeros(5,dtype = np.int8) #Same comment 
    current_in_neighbours_vec = np.ones(5,dtype=np.int8) #Same comment
    #Now I want to get the list of the indices of current_next_states in neighbours.
    for direction in np.arange(5):
        (current_next_state_indx_array,) = np.where(np.all(neighbours == current_next_states[direction,:], axis = 1))
        if len(current_next_state_indx_array) == 0:
            current_in_neighbours_vec[direction] = 0
        else:
            current_next_states_indx_vec[direction] = current_next_state_indx_array[0]
   

    U_temp = U_start.copy()
    U = U_start.copy()

    #First depth-1 steps (minimum expected risk action)
    #Saves the minimum expected risk action for each state.
    min_actions = np.zeros((depth-1,num_neighbours),dtype=np.int8)
    for d in np.arange(depth-1):
        for neigh in np.arange(num_neighbours):
            sums = np.zeros((5,5))
            for direction in np.arange(5):
                next_state = next_state_vec[neigh,direction]
                out_of_bounds = out_of_bounds_vec[neigh,direction] 
                next_state_indx = next_state_indx_vec[neigh,direction]
                in_neighbours = in_neighbours_vec[neigh,direction]
                
                sums[:,direction]=discount*ps[neighbours[neigh,0],neighbours[neigh,1],:,direction]*U[next_state_indx]*(out_of_bounds == 0)*in_neighbours

            action_risks = np.sum(sums,1)
            chosen_action = np.argmin(action_risks) 
            U_temp[neigh] = action_risks[chosen_action]
            min_actions[d,neigh] = chosen_action
        
        U = U_temp.copy()
        #Ensure U is 1 for unsafe states.
        U[unsafe_states] = 1
        U_temp[unsafe_states] = 1

    #Final Step (fixed action) (just current state)
    sums1=discount*ps[current_state[0],current_state[1],:,:]*U[current_next_states_indx_vec[:]]*(current_out_of_bounds == 0)*current_in_neighbours_vec              
    U_s_a = np.sum(sums1,1)

    variance_U = risk_variance(current_state,neighbours,unsafe_states,num_neighbours,depth,ps,cov,min_actions,U_s_a,out_of_bounds_vec,next_state_indx_vec,in_neighbours_vec,current_out_of_bounds,current_next_states_indx_vec,current_in_neighbours_vec,U_start)
    return U_s_a,variance_U
