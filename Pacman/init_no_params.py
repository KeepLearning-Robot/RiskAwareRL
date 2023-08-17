import numpy as np
import networkx as nx

#One param which we keep constant
#from 0 to 1, what is the probability of ghosts to go towards pacman at each iteration 
ghosts_determinism=0.9 

#Some Necessary Initializations
layout_mat = np.loadtxt('layout_small.txt',delimiter=',',dtype=np.int16)
#Locations in the layout
wall_key=8
ghost_key=7
pac_man_key=6
food1_key=1
food2_key=2
empty_space_key=0

#Initial Locations
ghost_location = np.where(layout_mat == ghost_key)
g_start = [ghost_location[0][0],ghost_location[1][0]]
pac_start = [np.where(layout_mat == pac_man_key)[0][0],np.where(layout_mat == pac_man_key)[1][0]]

wall_mat = layout_mat == wall_key
X_limit=np.size(layout_mat,0)
Y_limit=np.size(layout_mat,1)

total_number_of_actions=4
total_number_of_directions=25 #Because it is possible (in the event of hitting a wall) (To stay still)

#u_val is 0 safe, 1 for first food, 2 for second food, 4 for a ghost
neutral=0
food1=1
food2=2
target=3
ghost=4

def layout_to_u(lay):
    if lay == ghost_key or lay == pac_man_key:
        u_val = neutral
    else:
        u_val = lay
    return u_val

u_mat = np.vectorize(layout_to_u)(layout_mat)


##Location should be a matrix containing locations, assuming the layout has walls along the edges. 
def location2state(location,Y_movable_limit): 
    return location[0]*Y_movable_limit + location[1]

def state2location(state,Y_movable_limit): #
    rows = state // Y_movable_limit
    cols = state % Y_movable_limit
    return np.array([rows,cols])


##Creating MDP graph
G=np.array([[1,1]])
wall_mat_temp=np.copy(wall_mat)
adj = np.array([[-1, 0], [0, -1], [0, 0], [1, 0], [0, 1]]) #To get the 5 adjacent states of a given state
for state in range(X_limit*Y_limit):
    location=state2location(state,Y_limit)
    if not wall_mat_temp[location[0],location[1]]:
        for add in adj:
            next_location = location + add
            if not wall_mat_temp[next_location[0],next_location[1]]:
                G = np.append(G,[[state, location2state(next_location,Y_limit)]],0)
                wall_mat_temp[location[0],location[1]]=1
G = G[1:,:]

G_network = nx.from_edgelist(G)
