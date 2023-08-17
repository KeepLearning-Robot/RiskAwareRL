import numpy as np
from init_no_params import *

def Q_r(aut_now, aut_next):
    if aut_next==3:
        r=1
    #Alternative! Immediately Increased Successes to 376 lol. From 0.
    #elif (aut_next==1 or aut_next==2) and aut_now==0:
    #    r=0.5
    else:
        r=0
    return r

def automaton(u_val,last_automaton_state):
    #0 safe, 1 is first food, 2 is second food, 3 is both food (target), 4 is unsafe
    if last_automaton_state == neutral:
        next_automaton_state = u_val
    elif last_automaton_state == food1:
        if u_val == neutral:
            next_automaton_state = food1
        elif u_val == food1:
            next_automaton_state = food1
        elif u_val == food2:
            next_automaton_state = target
        elif u_val == ghost:
            next_automaton_state = ghost
    elif last_automaton_state == food2:           
        if u_val == neutral:
            next_automaton_state = food2
        elif u_val == food1:
            next_automaton_state = target
        elif u_val == food2:
            next_automaton_state = food2
        elif u_val == ghost:
            next_automaton_state = ghost
    else: #We were already either unsafe or target
        next_automaton_state = last_automaton_state
    return next_automaton_state

#Calculate the u value based on the current and next state
def u_val_func(pac_next,g_next):
    u_val = u_mat[pac_next[0],pac_next[1]]
    if (pac_next == g_next).all():
        u_val = ghost 
    return u_val

def action_to_implied_direction(action_indx):
    action_indx_as_dir = action_indx
    if action_indx >= 2:
        action_indx_as_dir += 1
    return action_indx_as_dir

def move_agent_direction(agent_location,direction_indx):
    next_location = agent_location.copy()
    if direction_indx == 0: #Right
        next_location[1] = agent_location[1]+1
    elif direction_indx == 1: #Up
        next_location[0] = agent_location[0]-1
    #Else 2, means stay!
    elif direction_indx == 3: #Down
        next_location[0] = agent_location[0]+1
    elif direction_indx == 4: #Left
        next_location[1] = agent_location[1]-1

    #Check if we tried to go into a wall
    hit_wall = wall_mat[next_location[0],next_location[1]]
    if hit_wall:
        next_location = agent_location.copy()
    return next_location,hit_wall

#Calculate next state given directions (and if we tried to go out of bounds)
def take_direction_wall(current_location,direction):
    pac_curr = current_location[0:2]
    g_curr = current_location[2:4]
    pacdirection = direction // 5
    gdirection = direction % 5
    pac_next,pac_wall = move_agent_direction(pac_curr,pacdirection)
    g_next,g_wall = move_agent_direction(g_curr,gdirection)
    aut = automaton(u_val_func(pac_next,g_next),current_location[4])
    return np.concatenate((pac_next,g_next,[aut])), pac_wall or g_wall

#Calculate next state given directions (and if we tried to go out of bounds)
def take_direction(current_location,direction):
    pac_curr = current_location[0:2]
    g_curr = current_location[2:4]
    pacdirection = direction // 5
    gdirection = direction % 5
    pac_next,_ = move_agent_direction(pac_curr,pacdirection)
    g_next,_ = move_agent_direction(g_curr,gdirection)
    aut = automaton(u_val_func(pac_next,g_next),current_location[4])
    return np.concatenate((pac_next,g_next,[aut]))

def movement_to_direction(previous,next):
    movement = next - previous
    #Right, Up, Stay, Down, Left
    if movement[1]==1:
        direction=0
    elif movement[0]==-1:
        direction=1
    elif movement[0]==1:
        direction=3
    elif movement[1]==-1:
        direction=4
    else:
        #Stay
        direction=2
    return direction

def chase_pacman(pac_next,g_cur):
    pac_next_state = location2state(pac_next,Y_limit)
    g_state = location2state(g_cur,Y_limit)
    if (pac_next == g_cur).all():
        g_next = g_cur.copy()
    else:
        g_next_state = nx.shortest_path(G_network,g_state,pac_next_state)[1]
        g_next = state2location(g_next_state,Y_limit)
    direction_taken = movement_to_direction(g_cur,g_next)
    return g_next, direction_taken

#Take an action, record the direction taken (by the ghost)
def take_action_direction(current_location,action_indx,is_det):
    #Right, Up, Stay, Down, Left
    pac_dir = action_to_implied_direction(action_indx)
    pac_curr = current_location[0:2]
    pac_next,pac_wall = move_agent_direction(pac_curr,pac_dir)
    if pac_wall:
        pac_dir = 2 #Then PacMan Stayed
    g_curr = current_location[2:4]
    if (not is_det) and (np.random.uniform() > ghosts_determinism):
        g_direction = int(5*np.random.uniform())
        g_next,g_direction = move_agent_direction(g_curr,g_direction)
    else: #Ghost chases pacman
            g_next,g_direction = chase_pacman(pac_next,g_curr)
    aut = automaton(u_val_func(pac_next,g_next),current_location[4])
    #THIS IS THE PROBLEM!
    direction_taken = g_direction + 5*pac_dir
    return np.concatenate((pac_next,g_next,[aut])), direction_taken

#For vectorizing local_U_update_fixed_actions
#@profile
def take_directions_wall(current_location):
    hit_wall = np.zeros(total_number_of_directions)
    next_locations = np.tile(current_location, (total_number_of_directions,1)) #Will default to current if we hit a wall, doesn't matter.

    #Is this the problem?
    dir_to_movement = np.array([[0,1],[-1,0],[0,0],[1,0],[0,-1]])
    dir_to_movement_2 = np.array([np.concatenate((a,b)) for a in dir_to_movement for b in dir_to_movement])
    next_locations[:,0:4] += dir_to_movement_2[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
    #Couldnt figure out a good way around this loop
    for dir in np.arange(total_number_of_directions):
        attempted_state = next_locations[dir,0:5]
        if wall_mat[attempted_state[0],attempted_state[1]] or wall_mat[attempted_state[2],attempted_state[3]]:
            next_locations[dir,0:5] = current_location #Doesn't matter, but should be in neighbours.
            hit_wall[dir] = True
        else: #Because automaton state doesn't matter if we hit a wall.
            next_locations[dir,4] = automaton(u_val_func(next_locations[dir,0:2],next_locations[dir,2:4]),current_location[4])
    return next_locations,hit_wall

#Local risk calcuation with fixed actions
#@profile
def local_U_update_fixed_actions(current_state,neighbours,unsafe_states,depth,ps_delta,min_actions,hit_walls_vec,next_state_indx_vec,in_neighbours_vec,current_hit_walls,current_next_states_indx_vec,current_in_neighbours_vec,U_start):
    discount = 1
    U = U_start.copy()

    #First depth-1 steps at once, with actions given by min_actions
    for d in np.arange(depth-1):
        actions = min_actions[d,:]
        sums=discount*ps_delta[neighbours[:,0],neighbours[:,1],neighbours[:,2],neighbours[:,3],actions,:]*U[next_state_indx_vec[:,:]]*(hit_walls_vec[:,:] == 0)*in_neighbours_vec[:,:]                
        U = np.sum(sums, axis = 1)

        #Ensure U is 1 for unsafe states.
        U[unsafe_states] = 1
   
    #FINAL STEP (fixed all actions)        
    sums1=discount*ps_delta[current_state[0],current_state[1],current_state[2],current_state[3],:,:]*U[current_next_states_indx_vec[:]]*(current_hit_walls == 0)*current_in_neighbours_vec              
    
    U_delta = np.sum(sums1,1)
    return U_delta

#Variance on Expected Risk Calculation
#@profile
def risk_variance(current_state,neighbours,unsafe_states,num_neighbours,depth,ps,cov,min_actions,U_s_a,hit_walls_vec,next_state_indx_vec,in_neighbours_vec,current_hit_walls,current_next_states_indx_vec,current_in_neighbours_vec,U_start):
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
    variance_U = np.zeros(total_number_of_actions) 
        
    #Now calculate variance
    #Do simultaneously for each first_action
    for neigh in np.arange(num_neighbours):
        for a in np.arange(total_number_of_actions):
            #For storing gradients
            grad = np.zeros((total_number_of_directions,total_number_of_actions))
            for d in np.arange(total_number_of_directions):
                #Calculate derivative wrt. p_neigh_a_d
                ps_delta = ps.copy()
                ps_delta[neighbours[neigh,0],neighbours[neigh,1],neighbours[neigh,2],neighbours[neigh,3],a,d] = \
                    ps_delta[neighbours[neigh,0],neighbours[neigh,1],neighbours[neigh,2],neighbours[neigh,3],a,d] + delta

                #Replicate local_U_update calculation but with actions
                #according to min_actions.
                '''HERE'''

                U_delta = local_U_update_fixed_actions(current_state,neighbours,unsafe_states,depth,ps_delta,min_actions,hit_walls_vec,next_state_indx_vec,in_neighbours_vec,current_hit_walls,current_next_states_indx_vec,current_in_neighbours_vec,U_start)
                #Row? vector of gradients for risk after each first_action, wrt
                #p_neigh_a_d.
                #Each column contains the gradients wrt p_neigh_a_allds for the
                #risk after a particular action
                grad[d,:] = (U_delta - U_s_a)/delta
            
            
            #Add contribution to variance. @ is matrix multiplication
            for first_action in np.arange(total_number_of_actions):
                variance_U[first_action] = variance_U[first_action] + np.transpose(grad[:,first_action]) @ cov[neighbours[neigh,0],neighbours[neigh,1],neighbours[neigh,2],neighbours[neigh,3],a,:,:] @ grad[:,first_action]
    return variance_U

#Local Risk Calculation
#@profile
def local_U_update(current_state,neighbours,depth,ps,cov):
    discount = 1
    num_neighbours = np.shape(neighbours)[0]

    #Pre-Calculate as Much as We Can.
    U_start = np.zeros(num_neighbours)
    unsafe_states = neighbours[:,4] == 4
    U_start[unsafe_states] = 1
    
    #Pre-calculate next_state, next_state_indx, out_of_bounds and in_neighbours
    #for each neigh and direction
    next_state_vec = np.zeros((num_neighbours,total_number_of_directions,5),dtype=np.int16)
    hit_wall_vec = np.zeros((num_neighbours,total_number_of_directions),np.int16)
    next_state_indx_vec = np.zeros((num_neighbours,total_number_of_directions),np.int16) #Defaults to 0 if not in neighbours. Won't influence calculation
    in_neighbours_vec = np.ones((num_neighbours,total_number_of_directions),np.int16)
    for neigh in np.arange(num_neighbours):
        for direction in np.arange(total_number_of_directions):
            next_state,hit_wall = take_direction_wall(neighbours[neigh],direction)
            next_state_vec[neigh,direction] = next_state
            hit_wall_vec[neigh,direction] = hit_wall
            
            (next_state_indx_array,) = np.where(np.all(neighbours == next_state, axis = 1))

            if len(next_state_indx_array) == 0:
                in_neighbours_vec[neigh,direction] = 0
                #next_state_indx_vec defaults to 0
            else:
                #in_neighbours defaults to 1
                next_state_indx_vec[neigh,direction] = next_state_indx_array[0]
    #Also pre-calculate next_states, next_states_indx, out_of_bounds and in_neighbours
    #For the final step, with CURRENT STATE and All Directions
    current_next_states,current_hit_walls = take_directions_wall(current_state) #Note that out_of_bounds here is 1 dimensional, but is used in the multiplication as 2 dimensional, because it is the same over actions.
    current_next_states_indx_vec = np.zeros(total_number_of_directions,dtype = np.int16) #Same comment 
    current_in_neighbours_vec = np.ones(total_number_of_directions,dtype=np.int16) #Same comment
    #Now I want to get the list of the indices of current_next_states in neighbours.
    for direction in np.arange(total_number_of_directions):
        (current_next_state_indx_array,) = np.where(np.all(neighbours == current_next_states[direction,:], axis = 1))
        if len(current_next_state_indx_array) == 0:
            current_in_neighbours_vec[direction] = 0
        else:
            current_next_states_indx_vec[direction] = current_next_state_indx_array[0]
   
    U_temp = U_start.copy()
    U = U_start.copy()
    #print('U_start: {U}'.format(U=U))

    #First depth-1 steps (minimum expected risk action)
    #Saves the minimum expected risk action for each state.
    min_actions = np.zeros((depth-1,num_neighbours),dtype=np.int16)
    for d in np.arange(depth-1):
        for neigh in np.arange(num_neighbours):
            sums = np.zeros((total_number_of_actions,total_number_of_directions))
            for direction in np.arange(total_number_of_directions):
                next_state = next_state_vec[neigh,direction]
                out_of_bounds = hit_wall_vec[neigh,direction] 
                next_state_indx = next_state_indx_vec[neigh,direction]
                in_neighbours = in_neighbours_vec[neigh,direction]
                
                sums[:,direction]=discount*ps[neighbours[neigh,0],neighbours[neigh,1],neighbours[neigh,2],neighbours[neigh,3],:,direction]*U[next_state_indx]*(out_of_bounds == 0)*in_neighbours

            action_risks = np.sum(sums,1)
            chosen_action = np.argmin(action_risks) 
            U_temp[neigh] = action_risks[chosen_action]
            min_actions[d,neigh] = chosen_action
        
        U = U_temp.copy()
        #Ensure U is 1 for unsafe states.
        U[unsafe_states] = 1
        U_temp[unsafe_states] = 1
        #print('U_{depth}: {U}'.format(depth=d,U=U))


    #Final Step (fixed action) (just current state)
    sums1=discount*ps[current_state[0],current_state[1],current_state[2],current_state[3],:,:]*U[current_next_states_indx_vec[:]]*(current_hit_walls == 0)*current_in_neighbours_vec              
    U_s_a = np.sum(sums1,1)

    #print('Current_Next_states_indx_vec: {next_states}'.format(next_states=current_next_states_indx_vec))


    variance_U = risk_variance(current_state,neighbours,unsafe_states,num_neighbours,depth,ps,cov,min_actions,U_s_a,hit_wall_vec,next_state_indx_vec,in_neighbours_vec,current_hit_walls,current_next_states_indx_vec,current_in_neighbours_vec,U_start)
    return U_s_a,variance_U
