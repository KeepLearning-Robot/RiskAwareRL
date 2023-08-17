from save_ragged import *
import numpy as np
import os
from initialization import *
import matplotlib.pyplot as plt
import sys

params_as_arguments = True
if params_as_arguments:
    critical_threshold =  sys.argv[1]
    choice_of_C = sys.argv[2]
    experiment_min = int(sys.argv[3])
    experiment_max = int(sys.argv[4])
    prior_choice = sys.argv[5]

#LoadFilePath Initializations
basepath = os.getcwd()
results_path = os.path.join(basepath,'Results')

#Load Data for Single Run and Add it to Accumulators
for experiment_number in range(experiment_min, experiment_max + 1):
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

    successLengths = pathLengths * successful_runs + max_iterations * (1 - successful_runs)
    print(len(successLengths))
    #Accumulate Data
    #Initialize Accumulators for the Data we want to graph
    if experiment_number == experiment_min:
        totalSuccessLengths = successLengths
        totalFail_history = fail_history
        totalSuccess_history = success_history
        totalQ_maxes = Q_maxes
    #Add Data to Accumulators
    else:
        totalSuccessLengths += successLengths
        totalFail_history += fail_history
        totalSuccess_history += success_history
        totalQ_maxes += Q_maxes

averageSuccessLengths = totalSuccessLengths / (experiment_max + 1 - experiment_min)
averageFail_history = totalFail_history / (experiment_max + 1 - experiment_min) 
averageSuccess_history = totalSuccess_history / (experiment_max + 1 - experiment_min) 
averageQ_maxes = totalQ_maxes / (experiment_max + 1 - experiment_min) 

plots_path = os.path.join(basepath,'PaperPlots')

averageExperiment_name = "bridge_Pmax{Pmax}_C{C}_Prior{prior}_averageExperiment{number1}to{number2}".format( \
        Pmax = critical_threshold,prior = prior_choice[0:4],C = choice_of_C,number1=experiment_min,number2=experiment_max)

plt.figure()
plt.title("prior 1, p\u2098\u2090\u2093 {pmax}".format(pmax=critical_threshold))
#plt.title("Q-Learning Only")
plt.ylabel("average steps taken to cross successfully")
plt.xlabel("episode number")
plt.plot(averageSuccessLengths, label = 'data')
plt.plot(22*np.ones((len(averageSuccessLengths))),'--', label = 'optimal', )
plt.legend()
#plt.setp(plot1[1],)
plt.savefig(os.path.join(plots_path,averageExperiment_name + "_successLengths.png"))

print(averageExperiment_name)
print("Successes: {n}".format(n = averageSuccess_history[-1]))
print("Failures: {n}".format(n = averageFail_history[-1]))
print("Total: {n}".format(n = len(averageSuccess_history)))

#plt.figure()
#plt.plot(averageFail_history)
#plt.savefig(os.path.join(plots_path,averageExperiment_name + "_failHistory.png"))

#plt.figure()
#plt.matshow(averageQ_maxes)
#plt.savefig(os.path.join(plots_path,averageExperiment_name + "_Qfunction.png.png"))



#Reconstruct which runs were fails:
#failures = 0
#fail_runs = np.zeros(len(fail_history))
#for i,fails in enumerate(fail_history):
#    if fails > failures:
#        failures += 1
#        fail_runs[i] = 1
