import numpy as np
import matplotlib.pyplot as plt
import os
from init_no_params import *

basepath = os.getcwd()
plots_path = os.path.join(basepath,'Plots')

plt.figure()
plt.matshow(wall_mat, cmap='Blues')
plt.savefig(os.path.join(plots_path,"pacman_setup_diagram.png"))
