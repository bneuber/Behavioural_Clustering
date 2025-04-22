# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:09:45 2025

@author: Bastian
"""

# %% IMPORT LIBRARIES

import numpy as np
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt


# %% INPUT PARAMETERS

# %% INPUT PARAMETERS

# define raw data file path
RawData_path = "C:/MPIAB/Data/DTW/03D4_100.csv"
# define acc data coding mode
acc_modes = ['wildog', 'butterfly']
# define current acc mode
acc_mode = acc_modes[0]
# define acc burst column name
col_acc = 'accInGBurst'
# define initial shift
sampling_init_shift = 0
# define burst length
burst_length = 24
# define number of clusters
n_clusters = 9
# define SOM dimensions
SOM_rows, SOM_cols = 3, 3
# define learning rate
learning_rate = 0.1
# define neighbourhood radius
neighborhood_radius = 1


# %% INITIALIZATION

# derive number of nodes of SOM
SOM_nodes = SOM_cols * SOM_rows






n_rows, n_cols = 2, 2              # SOM-Gitter (2x2)
n_nodes = n_rows * n_cols
burst_length = 40                 # Zeitpunkte pro Burst
dimensions = 4                    # 4D-Vektoren
learning_rate = 0.1
neighborhood_radius = 1           # Einflussreichweite (in Gitterdistanz)
np.random.seed(42)
