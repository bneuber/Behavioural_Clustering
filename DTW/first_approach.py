# %% IMPORT

import pandas as pd
import numpy as np
from scipy.spatial import distance as dist


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
n_clusters = 10


# %% DEFINE FUNCTIONS

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

# function to calculate costs of vector signal alignment
def alignment_cost(x, y):
    # calculate costs
    dist_mat = dist.cdist(x, y, "cosine")
    # initialize path variable
    alignment_path = [(len(x) - 1, len(y) - 1)]
    # initialize step counters
    i, j = len(x) - 1, len(y) - 1
    # run while final destination of the path has not been reached
    while (i > 0 or j > 0):
        # if i has already reached 0
        if i == 0:
            # one step up
            j -= 1
        # if j has already reached 0
        elif j == 0:
            # one step left
            i -= 1
        # if neither i nor j has reached 0
        else:
            # minimum of three left up neighbours
            direction = np.argmin([
                dist_mat[i - 1, j],    # up
                dist_mat[i, j - 1],    # left
                dist_mat[i - 1, j - 1] # up left
            ])
            # if the minimum is found at a step to the first instance of direction
            if direction == 0:
                # do a step upwards
                i -= 1
            # if the minimum is found at a step to the second instance of direction
            elif direction == 1:
                # do a step to the left
                j -= 1
            # if the minimum is found at a step to the third instance of direction
            else:
                # do a step to the up left
                i -= 1  # up
                j -= 1  # left
        alignment_path.append((i, j))
    # sum up all distances along the optimal path and get total alignment costs
    total_cost = sum([dist_mat[alignment_path[k][0], alignment_path[k][1]] for k in range(len(alignment_path))])
    # return resulting alignment cost
    return total_cost

# function to update refrence distance matrix
def update_ref_dist_matrix(ref):
    # initialize matrix of reference burst differences
    ref_cost_matrix = np.empty((n_clusters, n_clusters))
    # run over all reference brusts as rows and columns of reference distance matrix
    for row in range(1, len(ref)):
        for col in range(row):
            # calculate distance of current reference bursts
            ref_cost_matrix[row, col] = alignment_cost(ref[row], ref[col])
    # return the matrix of all costs
    return ref_cost_matrix


# %% DERIVE CONTINOUOS ACC DATA FRAME FROM RAW DATA

print('derive contiuousdata from raw data...')
# load raw data
RawData = pd.read_csv(RawData_path)
# if acc raw data is coded as a string of all axes for each burst (wildog)
if acc_mode == 'wildog':
    # initialize list of data frames
    acc_list = []
    # run over all bursts
    for i in RawData[col_acc]:
        # call function to convert raw burst data to data frame for current burst
        current_df = raw2floatlist(i)
        # append current acc data to list of data frames
        acc_list.append(current_df)
    # merge all data frames
    acc_df = pd.concat(acc_list, ignore_index=True)
# if acc raw data is coded in a data frame
if acc_mode == 'butterfly':
    # get acc data frame
    acc_df = RawData[['accX_mg', 'accY_mg', 'accZ_mg']]


# %% INITIALIZE REFERENCE ARRAY

# devide acc data to bursts
acc_bursts = [acc_df.iloc[i:i + burst_length] for i in range(0, len(acc_df), burst_length)]
# initialize references
ref = acc_bursts[:n_clusters]
# get cost matrix of reference bursts
ref_cost_mat = update_ref_dist_matrix(ref)
# initialize clusters: [number of burst, number of cluster]
clusters = [[i, i, 0] for i in range(n_clusters)]
# initialize cluster numbers
cluster_numbers = list(range(n_clusters))


# %% TRAINING OF CLUSTERS

# run over all bursts of the raw data
for burst_index, crnt_burst in enumerate(acc_bursts):
    # get costs of all alignments with reference bursts
    crnt_costs = [alignment_cost(ref[ref_index], crnt_burst) for ref_index in range(n_clusters)]
    # get minimum of crnt_costs
    min_costs, cluster_found = min(crnt_costs), crnt_costs.index(min(crnt_costs))

