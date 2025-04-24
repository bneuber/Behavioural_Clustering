# %% IMPORT LIBRARIES

import numpy as np
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt
import pandas as pd


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
# set dimensions of raw data
dimensions = 3


# %% DEFINITION OF FUNCTIONS

# function to strip acc raw to list of float values
def raw2floatlist(raw_burst):
    # convert raw string to list of float numbers
    acc = np.array(list(map(float, raw_burst.split()))).tolist()
    # split to channels
    acc_df = pd.DataFrame({
        'acc_x': acc[0::3],
        'acc_y': acc[1::3],
        'acc_z': acc[2::3]
    })
    # return result
    return acc_df

# function to get grid position of SOM
def grid_position(index, SOM_cols):
    # returns two SOM indices of the current prototype
    return np.array([index // SOM_cols, index % SOM_cols])

# function to calculate neighbourhood strength
def neighborhood_strength(bmp_ind, neighbour_ind):
    # calculate euklidian distance of the best matching prototype and it's neighbour
    d = np.linalg.norm(grid_position(bmp_ind) - grid_position(neighbour_ind))
    # return the neighbourhood coefficient by calculation of h = e^(-d² / (2*nr²))
    return np.exp(-d**2 / (2 * neighborhood_radius**2))

# function to update one prototypes with new burst information
def update_prototype(prototype, burst, lr):
    # get the path of the cheapest fitting for current burst and prototype
    path, _ = dtw_path(prototype, burst)
    # create a copy of the prototype list
    updated = prototype.copy()
    # run over all positions in the fitting path
    for (i, j) in path:
        # add difference between the fitted vectors between burst and prototype to the prototype vectors
        updated[i] += lr * (burst[j] - prototype[i])
    # return the updated prototype
    return updated


# %% INITIALIZATION

# derive number of nodes of SOM
SOM_nodes = SOM_cols * SOM_rows
# select a seed for random intitial cluster prototypes
np.random.seed(42)
# initialization of prototype array
prototypes = [np.random.randn(burst_length, dimensions) for _ in range(SOM_nodes)]
# initialize list of all labels classified by the learning algorithm
classified_labels = []


# %% DERIVE CONTINOUOS ACC DATA FRAME FROM RAW DATA

print('derive contiuousdata from raw data...')
# load raw data
RawData = pd.read_csv(RawData_path)
# if acc raw data is coded as a string of all axes for each burst (wildog)
if acc_mode == 'wildog':
    # initialize list of data frames
    acc_list = []
    # run over all bursts
    for raw_burst_ind in RawData[col_acc]:
        # call function to convert raw burst data to data frame for current burst
        current_df = raw2floatlist(raw_burst_ind)
        # append current acc data to list of data frames
        acc_list.append(current_df)
    # merge all data frames
    acc_df = pd.concat(acc_list, ignore_index=True)
# if acc raw data is coded in a data frame
if acc_mode == 'butterfly':
    # get acc data frame
    acc_df = RawData[['accX_mg', 'accY_mg', 'accZ_mg']]


# %% LEARNING PHASE

# run over all lines of raw data
for burst_ind, burst in enumerate(len(acc_df)):
    
    # match current burst to prototypes - - - - - - - - - - - - - - - - - - - -
    # get all distances between the current burst and all prototypes
    distances = [dtw_path(p, burst)[1] for p in prototypes]
    # identify the index of the best matching prototype
    bmp_ind = np.argmin(distances)
    # store index of best matching prototype in classification list
    classified_labels.append(bmp_ind)
    
    # update prototypes - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 
    new_prototypes = []
    # run over all prototypes
    for prototype_ind, prototype in enumerate(prototypes):
        # get the neighbourhood strength by calculation of distances
        h = neighborhood_strength(bmp_ind, prototype_ind)
        # update the current prototype
        updated = update_prototype(prototype, burst, learning_rate * h)
        # store the 
        new_prototypes.append(updated)
    prototypes = new_prototypes