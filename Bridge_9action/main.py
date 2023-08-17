import matplotlib.pyplot as plt
from save_ragged import *
from datetime import datetime
import os
import shutil
from functions import *
from tqdm import tqdm
import time

"""
Parameter setup section
"""
pc = None
prior_choice = "Uninformative Prior"

if prior_choice == "Uninformative Prior":
    critical_threshold = 0.33
    # critical_threshold = 0.01
    pc = "1"
elif prior_choice == "Medium Informative Prior":
    critical_threshold = 0.01
    pc = "2"
elif prior_choice == "High Informative Prior":
    # critical_threshold = 0.0033
    critical_threshold = 0.01
    pc = "3"
else:
    raise Exception("NotImplemented")

# SaveFilePath Initializations
now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")
basepath = os.path.join(os.getcwd(),'runs', f"prior_{pc}", dt_string)
results_path = os.path.join(basepath, 'Results')
if not os.path.isdir(basepath):
    os.makedirs(basepath)
if not os.path.isdir(results_path):
    os.makedirs(results_path)
if not os.path.isdir(os.path.join(basepath, 'Plots')):
    os.makedirs(os.path.join(basepath, 'Plots'))
if not os.path.isdir(os.path.join(basepath, 'PaperPlots')):
    os.makedirs(os.path.join(basepath, 'PaperPlots'))
log = os.path.join(basepath, "log.txt")
printf(f"Current time: {dt_string}", log)

# Experiment Parameters
safe_padding = True  # make it True to activate padding
number_of_runs = 1500
max_iterations = 400

# Hyper Parameters
discount_factor_Q = 0.9
alpha = 0.85
observation_radius = 2
temp = .005

# Choice of Prior
choice_of_C = "ThesisDecreasing"

printf(f"\n#############################\n"
       f"safe_padding: {safe_padding}\n"
       f"number_of_runs: {number_of_runs}\n"
       f"max_iterations: {max_iterations}\n"
       f"discount_factor_Q: {discount_factor_Q}\n"
       f"alpha: {alpha}\n"
       f"observation_radius: {observation_radius}\n"
       f"temp: {temp}\n"
       f"prior_choice: {prior_choice}\n"
       f"critical_threshold: {critical_threshold}\n"
       f"choice_of_C: {choice_of_C}\n"
       f"#############################\n", log)

# experiment_number = sys.argv[1]
runs = 10
printf(f"\nYou want {runs} tests.", log)

save_runs = 100
printf(f"Breakpoint will be saved every {save_runs} steps", log)

shutil.copy(__file__, os.path.join(basepath, "main.py"))

printf("Starting Experiment...", log)
"""
END OF THIS SECTION
"""

file_start_num = 4

cnt = 1
for experiment_number in range(file_start_num, file_start_num+int(runs)):
    experiment_name = "bridge_Pmax{Pmax}_C{C}_Prior{prior}_experiment{number}".format(
        Pmax=critical_threshold, prior=prior_choice[0:4], C=choice_of_C, number=experiment_number)
    printf(f"\n-------- Experiment ({cnt} of {runs}) ---------\nName: {experiment_name}", log)
    cnt += 1

    # 20 by 20 grid
    X_movable_limit = 20
    Y_movable_limit = 20
    X = np.linspace(1, X_movable_limit, X_movable_limit)
    Y = np.linspace(1, Y_movable_limit, Y_movable_limit)
    [gridx, gridy] = np.meshgrid(X, Y)

    # State Labels
    u = np.zeros([20, 20])
    u[0:7, 0:20] = 2  # goal zone
    u[8:12, 0:8] = 1  # dangerous area
    u[8:12, 11:20] = 1  # dangerous area

    # Initialize Counters/Trackers
    it_run_num = 1
    number_of_fails = 0
    failed_path = []
    fail_history = []
    success_history = []
    number_of_successes = 0
    path_history = []
    successful_runs = np.zeros(number_of_runs)

    # Q function. A 4-dim matrix (action, state(X, Y), automaton_state)
    total_number_of_actions = 9
    Q = np.zeros((total_number_of_actions, X_movable_limit, Y_movable_limit, 3))
    # Is there a point having the automaton_state here? Maybe just for generality?

    # State Counts
    state_count = np.zeros((X_movable_limit, Y_movable_limit))
    state_count[19, 0] = 1  # initial state

    # posterior-prior = counts
    # Defining the Prior
    # Agent's belief about transition probability
    if prior_choice == "Uninformative Prior":
        state_action_direction_prior = np.zeros((X_movable_limit, Y_movable_limit,
                                                 total_number_of_actions, total_number_of_actions))
        for action_indx in np.arange(total_number_of_actions):
            for x in np.arange(0, X_movable_limit):
                for y in np.arange(0, Y_movable_limit):
                    for direction in np.arange(0, total_number_of_actions):
                        poss_next_state, out_of_bounds = take_action_m_boundary([x, y, 1], direction, True, u)
                        if out_of_bounds == False:
                            state_action_direction_prior[x, y, action_indx, direction] = 1
        state_action_direction_posterior = state_action_direction_prior

    if prior_choice == "Medium Informative Prior":
        state_action_direction_prior = np.zeros((X_movable_limit, Y_movable_limit,
                                                 total_number_of_actions, total_number_of_actions))
        for action_indx in np.arange(total_number_of_actions):
            for x in np.arange(0, X_movable_limit):
                for y in np.arange(0, Y_movable_limit):
                    for direction in np.arange(0, total_number_of_actions):
                        poss_next_state, out_of_bounds = take_action_m_boundary([x, y, 1], direction, True, u)
                        if out_of_bounds == False:
                            if action_indx == direction:
                                state_action_direction_prior[x, y, action_indx, direction] = 12
                            else:
                                state_action_direction_prior[x, y, action_indx, direction] = 1
        state_action_direction_posterior = state_action_direction_prior

    if prior_choice == "High Informative Prior":
        state_action_direction_prior = np.zeros((X_movable_limit, Y_movable_limit,
                                                 total_number_of_actions, total_number_of_actions))
        for action_indx in np.arange(total_number_of_actions):
            for x in np.arange(0, X_movable_limit):
                for y in np.arange(0, Y_movable_limit):
                    for direction in np.arange(0, total_number_of_actions):
                        poss_next_state, out_of_bounds = take_action_m_boundary([x, y, 1], direction, True, u)
                        if out_of_bounds == False:
                            if action_indx == direction:
                                state_action_direction_prior[x, y, action_indx, direction] = 96
                            else:
                                state_action_direction_prior[x, y, action_indx, direction] = 1
        state_action_direction_posterior = state_action_direction_prior

    # State_action_prior: similar to the above, based on the prior given above
    state_action_prior = np.sum(state_action_direction_prior, 3)
    state_action_posterior = state_action_prior  # posterior is initialized the same as prior

    # Expected transition probabilities
    ps = np.divide(state_action_direction_prior, np.expand_dims(state_action_prior, 3))

    # Covariance Matrix: cov(q1,q2,action,:,:) is the 5*5 covariance matrix for
    # the dirichlet for taking action at state (q1,q2)
    cov = np.zeros(
        (X_movable_limit, Y_movable_limit, total_number_of_actions, total_number_of_actions, total_number_of_actions))
    for q1 in np.arange(0, X_movable_limit):
        for q2 in np.arange(0, Y_movable_limit):
            for a in np.arange(0, total_number_of_actions):
                for direction_i in np.arange(0, total_number_of_actions):
                    for direction_j in np.arange(0, total_number_of_actions):
                        cov[q1, q2, a, direction_i, direction_j] = ((direction_i == direction_j) * ps[
                            q1, q2, a, direction_i] - ps[q1, q2, a, direction_i] * ps[q1, q2, a, direction_j]) \
                                                                   / (state_action_posterior[q1, q2, a] + 1)

    # Defining the parameter C as a function of state_count
    if choice_of_C == "ThesisDecreasing":
        def C_function(cur_state_count):
            return 0.7 * max((24 - cur_state_count) / 25, 0) + 0.3 / (1 + cur_state_count)

    if choice_of_C[0:5] == "Fixed":
        c = float(choice_of_C[5:9])


        def C_function(cur_state_cout):
            return c

    if choice_of_C[0:5] == "Slope":
        num = int(choice_of_C[5:7])


        def C_function(cur_state_count):
            return 0.7 * max(((num - 1) - cur_state_count) / num, 0) + 0.3 / (1 + cur_state_count)

    # Train Agent
    start_time = time.time()
    for run_num in tqdm(np.arange(1, number_of_runs + 1)):
        current_state = np.array([19, 0, 0])  # In the product MDP
        current_path = np.expand_dims(current_state, 0)  # Add a dimension so we can keep a list of states
        # printf('Run Number: ' + str(run_num), log)
        print(f"\nRun Number -> {run_num}")
        it_num = 1
        while (current_state[-1] != 2) and (it_num <= max_iterations):
            it_num = it_num + 1
            it_run_num = it_run_num + 1
            """
            product MDP: (x, y) as well as the automaton state (where automaton means if the robot is unsafe now)
            """

            if safe_padding:
                depth = 2  # Must be <= observation radius
                old_neighbours = np.expand_dims(current_state, 0)  # Add a dimension so we can keep a list of states
                neighbours = np.array([]).astype(np.int8).reshape(0, 3)
                # CAREFUL IF THE SIZE OF THE AREA IS LARGER THAN 128
                # Expand neighbours one at a time until covering the
                # Observation_radius: make sure it is unique.
                # NOTE: Neighbours is in the product MDP
                for i in np.arange(observation_radius):
                    for current_exploring in np.arange(np.size(old_neighbours, 0)):
                        for action in np.arange(total_number_of_actions):
                            neighbours = np.vstack(
                                [neighbours, take_action_m(old_neighbours[current_exploring], action, True, u)])
                        neighbours = np.unique(neighbours, axis=0)
                    old_neighbours = np.vstack([old_neighbours, neighbours])
                    old_neighbours = np.unique(old_neighbours, axis=0)

                # RISK AND VARIANCE CALCULATION!
                U, variance_U = local_U_update(current_state, neighbours, depth, u, ps, cov)

                # Defining the parameter C for risk-averseness based on state_count
                C = C_function(state_count[current_state[0], current_state[1]])

                # Perform Cantelli Bound calculation
                P_a = U + np.sqrt((variance_U * C) / (1 - C))
                (acceptable_actions,) = np.nonzero(P_a < critical_threshold)
                if acceptable_actions.size == 0:
                    # If nothing is acceptable, then just choose minimum U.
                    acceptable_actions = np.nonzero(U == np.amin(U))

            available_Qs = np.zeros(total_number_of_actions)
            if safe_padding:
                for each_action in acceptable_actions:
                    available_Qs[each_action] = Q[each_action, current_state[0], current_state[1], current_state[2]]
                for each_action in np.setdiff1d(np.arange(total_number_of_actions), acceptable_actions,
                                                assume_unique=True):
                    # Find the set difference of two arrays
                    available_Qs[each_action] = Q[each_action, current_state[0], current_state[1], current_state[
                        2]] - 300
            else:
                for each_action in np.arange(total_number_of_actions):
                    available_Qs[each_action] = Q[each_action, current_state[0], current_state[1], current_state[2]]

            # Boltzmann rational
            expo = np.exp(available_Qs / temp)
            probabs = expo / sum(expo)

            # Select an action
            actions = np.arange(total_number_of_actions)
            selected_action = np.random.choice(actions, p=probabs)

            # Take that action
            next_state, direction_taken = take_action_m_direction(current_state, selected_action, False, u)

            # UPDATES
            # Update posterior by adding counts
            state_action_direction_posterior[current_state[0], current_state[1], selected_action, direction_taken] = \
                state_action_direction_posterior[
                    current_state[0], current_state[1], selected_action, direction_taken] + 1

            state_action_posterior[current_state[0], current_state[1], selected_action] = \
                state_action_posterior[current_state[0], current_state[1], selected_action] + 1
            # Update Means
            ps[current_state[0], current_state[1], selected_action, :] = \
                state_action_direction_posterior[current_state[0], current_state[1], selected_action, :] / \
                state_action_posterior[current_state[0], current_state[1], selected_action]
            # Update Covariance Matrix
            for direction_i in np.arange(total_number_of_actions):
                for direction_j in np.arange(total_number_of_actions):
                    cov[current_state[0], current_state[1], selected_action, direction_i, direction_j] = ((
                                                                                                                  direction_i == direction_j) *
                                                                                                          ps[
                                                                                                              current_state[
                                                                                                                  0],
                                                                                                              current_state[
                                                                                                                  1], selected_action, direction_i] -
                                                                                                          ps[
                                                                                                              current_state[
                                                                                                                  0],
                                                                                                              current_state[
                                                                                                                  1], selected_action, direction_i] *
                                                                                                          ps[
                                                                                                              current_state[
                                                                                                                  0],
                                                                                                              current_state[
                                                                                                                  1], selected_action, direction_j]) \
                                                                                                         / (
                                                                                                                 state_action_posterior[
                                                                                                                     current_state[
                                                                                                                         0],
                                                                                                                     current_state[
                                                                                                                         1], selected_action] + 1)

            # Update Q function
            current_Qs = Q[:, next_state[0], next_state[1], next_state[2]]

            Q[selected_action, current_state[0], current_state[1], current_state[2]] = (1 - alpha) * \
                                                                                       Q[selected_action, current_state[
                                                                                           0],
                                                                                         current_state[1],
                                                                                         current_state[
                                                                                             2]] + alpha * \
                                                                                       (
                                                                                               Q_r(next_state) + discount_factor_Q * np.amax(
                                                                                           current_Qs))

            # Update state counts to reflect moving to next state
            state_count[next_state[0], next_state[1]] = state_count[next_state[0], next_state[1]] + 1
            current_path = np.vstack([current_path, next_state])

            # DISPLAY (AND TERMINATE IF FAILED)
            if next_state[2] == 1:
                number_of_fails = number_of_fails + 1
                # print('-------fail-------')
                # print('Current State: ' + str(current_state))
                # print('Next State: ' + str(next_state))
                # print('Neighbours: ' + str(neighbours))
                # print('U: ' + str(U))
                # print('Selected Action:' + str(selected_action))
                break
            elif next_state[2] == 2:
                # print('+++++++success+++++++')
                number_of_successes = number_of_successes + 1
                # Keep a record of which runs were successful.
                successful_runs[run_num - 1] = 1

            # Update current state
            current_state = next_state.copy()

        # Add run to paths.
        path_history.append(current_path)

        # Cumulative total of number of successes.
        fail_history.append(number_of_fails)
        success_history.append(number_of_successes)

        if run_num % int(save_runs) == 0:
            printf(f"\nSaving breakpoint @ experiment number <{experiment_number}> | run <{run_num}>", log)
            np.save(os.path.join(results_path, experiment_name + "_failHistory"), fail_history)
            np.save(os.path.join(results_path, experiment_name + "_successHistory"), success_history)
            np.save(os.path.join(results_path, experiment_name + "_successfulRuns"), successful_runs)
            np.save(os.path.join(results_path, experiment_name + "_Q"), Q)
            save_stacked_array(os.path.join(results_path, experiment_name + "_pathHistory"), path_history, axis=0)

    # Prints
    printf(f"Time Elapsed: {(time.time() - start_time):.2f} secs ({(time.time() - start_time) / 60:.2f} mins)", log)
    printf(f"Number of Successes: {str(number_of_successes)}", log)
    printf(f"Number of Failures: {str(number_of_fails)}", log)

    # Save Results!
    np.save(os.path.join(results_path, experiment_name + "_failHistory"), fail_history)
    np.save(os.path.join(results_path, experiment_name + "_successHistory"), success_history)
    np.save(os.path.join(results_path, experiment_name + "_successfulRuns"), successful_runs)
    np.save(os.path.join(results_path, experiment_name + "_Q"), Q)
    save_stacked_array(os.path.join(results_path, experiment_name + "_pathHistory"), path_history, axis=0)
    printf(f"\nTraining Completed.\nEnding experiment number <{experiment_number}>", log)

    # Plot
    # plt.plot(fail_history)
    # plt.show()
