import matplotlib.pyplot as plt
from save_ragged import *
import sys
import os
import numpy as np
from functions import *

experiment_min = 1
experiment_max = 10

qlearning = False

action_space_fold = "5action_runs"  # runs or 5action_runs

# fold = "bridge_Pmax0.33_CThesisDecreasing_PriorUnin_experiment"
# fold = "1500_bridge_Pmax0.01_CThesisDecreasing_PriorUnin_experiment"
# fold = "bridge_Pmax0.33_CThesisDecreasing_PriorMedi_experiment"
# fold = "bridge_Pmax0.01_CThesisDecreasing_PriorMedi_experiment"
# fold = "bridge_Pmax0.01_CThesisDecreasing_PriorMedi_experiment"
# fold = "bridge_Pmax0.01_CThesisDecreasing_PriorHigh_experiment"
# fold = "bridge_Pmax0.0033_CThesisDecreasing_PriorHigh_experiment"

# fold = "bridge_5action_Pmax0.33_CThesisDecreasing_PriorUnin_experiment"
# fold = "1500_bridge_5action_Pmax0.01_CThesisDecreasing_PriorUnin_experiment"
fold = "bridge_5action_Pmax0.33_CThesisDecreasing_PriorMedi_experiment"
# fold = "bridge_5action_Pmax0.01_CThesisDecreasing_PriorMedi_experiment"
# fold = "bridge_5action_Pmax0.01_CThesisDecreasing_PriorHigh_experiment"
# fold = "bridge_5action_Pmax0.0033_CThesisDecreasing_PriorHigh_experiment"
# fold = "QLearning_Only"

pc = None
# prior_choice = "Uninformative Prior"
prior_choice = "Medium Informative Prior"
# prior_choice = "High Informative Prior"

if prior_choice == "Uninformative Prior":
    critical_threshold = 0.01
    # critical_threshold = 0.33
    pc = "1"
elif prior_choice == "Medium Informative Prior":
    critical_threshold = 0.33
    # critical_threshold = 0.01
    pc = "2"
elif prior_choice == "High Informative Prior":
    critical_threshold = 0.0033
    # critical_threshold = 0.01
    pc = "3"
else:
    raise Exception("NotImplemented")

if not qlearning:
    basepath = os.path.join(os.getcwd(), action_space_fold, f"prior_{pc}")
    basepath = os.path.join(basepath, fold)
else:
    basepath = os.path.join(os.getcwd(), action_space_fold, fold)
results_path = os.path.join(basepath, 'Results')
plots_path = os.path.join(os.getcwd(), action_space_fold, 'PaperPlots')
if not os.path.isdir(plots_path):
    os.makedirs(plots_path)
# Experiment Parameters
safe_padding = True  # make it True to activate padding
number_of_runs = 500
max_iterations = 400

# Hyper Parameters
discount_factor_Q = 0.9
alpha = 0.85
observation_radius = 2
temp = .005
# Choice of Prior
choice_of_C = "ThesisDecreasing"

if action_space_fold == "runs":
    total_number_of_actions = 9
elif action_space_fold == "5action_runs":
    total_number_of_actions = 5

if not qlearning:
    print(f"\n#############################\n"
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
          f"#############################\n")

for experiment_number in range(experiment_min, experiment_max + 1):

    # experiment_index = [1,2,3,6,7,8]
    # experiment_min = 1
    # experiment_max = 8
    # for experiment_number in experiment_index:

    if action_space_fold != "5action_runs":
        experiment_name = "bridge_Pmax{Pmax}_C{C}_Prior{prior}_experiment{number}".format(
            Pmax=critical_threshold, prior=prior_choice[0:4], C=choice_of_C, number=experiment_number)
    else:
        experiment_name = "bridge_5action_Pmax{Pmax}_C{C}_Prior{prior}_experiment{number}".format(
            Pmax=critical_threshold, prior=prior_choice[0:4], C=choice_of_C, number=experiment_number)

    print(experiment_name)

    #############################################################################################

    # Defining the area
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

    relevant_Q = Q[:, :, :, 0]
    Q_maxes = np.max(relevant_Q, axis=0)

    # Reconstruct which runs were fails:
    failures = 0
    fail_runs = np.zeros(len(fail_history))
    for i, fails in enumerate(fail_history):
        if fails > failures:
            failures += 1
            fail_runs[i] = 1

    # Load Data
    fail_history = np.load(os.path.join(results_path, experiment_name + "_failHistory.npy"))
    success_history = np.load(os.path.join(results_path, experiment_name + "_successHistory.npy"))
    successful_runs = np.load(os.path.join(results_path, experiment_name + "_successfulRuns.npy"))
    Q = np.load(os.path.join(results_path, experiment_name + "_Q.npy"))
    pathHistory = load_stacked_arrays(os.path.join(results_path, experiment_name + "_pathHistory.npz"), axis=0)
    pathLengths = [len(path) for path in pathHistory]

    # successLengths = pathLengths * successful_runs
    successLengths = pathLengths * successful_runs + max_iterations * (1 - successful_runs)
    # print(len(successLengths))

    if experiment_number == 1:
        totalSuccessLengths = successLengths
        totalFail_history = fail_history
        totalSuccess_history = success_history
        totalQ_maxes = Q_maxes
    # Add Data to Accumulators
    else:
        totalSuccessLengths += successLengths
        totalFail_history += fail_history
        totalSuccess_history += success_history
        totalQ_maxes += Q_maxes

averageSuccessLengths = totalSuccessLengths / (experiment_max + 1 - experiment_min)
averageFail_history = totalFail_history / (experiment_max + 1 - experiment_min)
averageSuccess_history = totalSuccess_history / (experiment_max + 1 - experiment_min)
averageQ_maxes = totalQ_maxes / (experiment_max + 1 - experiment_min)

pc = None
if prior_choice == "Uninformative Prior":
    pc = "1"
elif prior_choice == "Medium Informative Prior":
    pc = "2"
elif prior_choice == "High Informative Prior":
    pc = "3"

averageExperiment_name = f"bridge_Pmax{critical_threshold}_C{choice_of_C}_Prior{prior_choice[0:4]}_averageExperiment{experiment_min}to{experiment_max}"
plt.figure()
if not qlearning:
    plt.title(f"prior {pc}, p\u2098\u2090\u2093 {critical_threshold}")
else:
    plt.title("Q-Learning Only")
plt.ylabel("average steps taken to cross successfully")
plt.xlabel("episode number")
plt.plot(averageSuccessLengths, label='data')
plt.plot(22 * np.ones((len(averageSuccessLengths))), '--', label='optimal', )
plt.legend()
# plt.setp(plot1[1],)
if not qlearning:
    plt.savefig(os.path.join(plots_path, averageExperiment_name + "_successLengths.png"))
else:
    plt.savefig(os.path.join(plots_path, "QL_successLengths.png"))

print(averageExperiment_name)
print("Successes: {n}".format(n=averageSuccess_history[-1]))
print("Failures: {n}".format(n=averageFail_history[-1]))
print("Total: {n}".format(n=len(averageSuccess_history)))
