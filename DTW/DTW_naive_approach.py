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
n_clusters = 8
# define reference cluster update criteria
ref_cluster_update_crit = ['cheapest ref',
                           'worst performance',
                           'cheapest unused ref'
                           ]
ref_cluster_update_crit = ref_cluster_update_crit[2]


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
    ref_cost_mat = np.empty((n_clusters, n_clusters))
    # get indices of upper right of referece cost matrix
    i, j = np.triu_indices(ref_cost_mat.shape[0], k=0)
    # set all upper right values of the matrix to infinity 
    ref_cost_mat[i, j] = np.nan
    # run over all reference brusts as rows and columns of reference distance matrix
    for row in range(1, len(ref)):
        for col in range(row):
            # calculate distance of current reference bursts
            ref_cost_mat[row, col] = alignment_cost(ref[row], ref[col])
    # return the matrix of all costs
    return ref_cost_mat

# function to update cluszters in case a new burst can be used as new reference burst
def update_clusters(ref, clusters, cluster_performance, ref_cost_mat, ref_cluster_update_crit, crnt_burst, burst_index, cluster_counter, burst_clusters):
    # if cheapest reference burst method is used to update clusters
    if ref_cluster_update_crit == "cheapest ref":
        # get the indices of cheapest alignment
        min_cost_index = np.unravel_index(np.nanargmin(ref_cost_mat), ref_cost_mat.shape)
        # initialize list for cost sums
        cost_sum = [0, 0]
        # run over both of the reference bursts of minimum alignment costs
        for ref_burst_ind in range(2):
            # get sum of costs for the first reference burst
            cost_sum[ref_burst_ind] = np.nansum(ref_cost_mat, axis=1)[min_cost_index[ref_burst_ind]] + np.nansum(ref_cost_mat, axis=0)[min_cost_index[ref_burst_ind]]
        # get the index of the cheapest reference burst
        replaced_ref_burst_index = min_cost_index[cost_sum.index(min(cost_sum))]
    # if the worst performance method is used to update clusters
    elif ref_cluster_update_crit == "worst performance":
        # if cluster 0 has been used for alignment
        if cluster_performance.loc[0, 'alignment counter'] != 1:
            # filter cluster performance data frame and identify the cluster of worst performance
            replaced_ref_burst_index = cluster_performance[cluster_performance['cluster no.'].isin([c[1] for c in clusters])]['avr cost'].idxmin()
            # replace worst performing cluster by last candidate
            ref[replaced_ref_burst_index] = ref[0]
            # replace worst performing cluster by last candidate
            clusters[replaced_ref_burst_index] = clusters[0]
        # set replacement index for position of new reference burst candidate (0)
        replaced_ref_burst_index = 0
    # if the cheapest unused reference burst method is used
    elif ref_cluster_update_crit == "cheapest unused ref":
        # get the indices of cheapest alignment
        min_cost_index = np.unravel_index(np.nanargmin(ref_cost_mat), ref_cost_mat.shape)
        # get cluster index of most unused cluster out of the cheapest ones
        replaced_ref_burst_index = cluster_performance[cluster_performance['cluster no.'].isin(min_cost_index)]['alignment counter'].idxmin()
    # replace cheapest reference burst by current burst
    ref[replaced_ref_burst_index] = crnt_burst
    # update cost matrix of reference bursts
    ref_cost_mat = update_ref_dist_matrix(ref)
    # update clusters
    clusters[replaced_ref_burst_index] = [burst_index, cluster_counter]
    # save cluster alignment of current burst
    burst_clusters.loc[burst_index] = burst_index, cluster_counter, 0
    # update cluster performance
    cluster_performance.loc[cluster_counter] = cluster_counter, burst_index, 1, 0
    # update cluster counter
    cluster_counter += 1
    # return results
    return ref, clusters, cluster_performance, burst_clusters, cluster_counter, ref_cost_mat
        

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
# initialize Data Frame of references
ref = acc_bursts[:n_clusters]
# initialize list of cluster numbers
cluster_index = list(range(n_clusters))
# get cost matrix of reference bursts
ref_cost_mat = update_ref_dist_matrix(ref)
# initialize clusters: [number of burst, number of cluster]
clusters = [[i, i] for i in range(n_clusters)]


# %% INITIALIZATION

# initialize cluster numbers
cluster_numbers = list(range(n_clusters))
# initialize cluster counter
cluster_counter = n_clusters
# initialize burst to cluster alignment
burst_clusters = pd.DataFrame(columns=['burst_index', 'cluster', 'min_costs'])
# initialize cluster performance array
cluster_performance = pd.DataFrame([[i, i, 0, 0] for i in range(n_clusters)], columns=['cluster no.', 'burst', 'alignment counter', 'avr cost'])


# %% TRAINING OF CLUSTERS

# run over all bursts of the raw data
for burst_index, crnt_burst in enumerate(acc_bursts):
    if burst_index == 4000:
        Halt = 1
    # get costs of all alignments with reference bursts
    crnt_costs = [alignment_cost(ref[ref_index], crnt_burst) for ref_index in range(n_clusters)]
    # # if cheapest reference burst method is used to update clusters
    # if ref_cluster_update_crit == 'cheapest ref':
    # if the minimum of current alignment costs is higher than alignment costs of all reference bursts
    if min(crnt_costs) > np.nanmin(ref_cost_mat):
        # update reference bursts and clusers
        ref, clusters, cluster_performance, burst_clusters, cluster_counter, ref_cost_mat = update_clusters(ref,
                                                                                                            clusters,
                                                                                                            cluster_performance,
                                                                                                            ref_cost_mat,
                                                                                                            ref_cluster_update_crit,
                                                                                                            crnt_burst,
                                                                                                            burst_index,
                                                                                                            cluster_counter,
                                                                                                            burst_clusters
                                                                                                            )
    # if the current_alignment costs ar not higher than lowest alignment costs within all reference bursts
    else:
        # get index of currently aligned cluster
        crnt_cluster_index = crnt_costs.index(min(crnt_costs))
        # get number of currently alligned cluster
        crnt_cluster_number = clusters[crnt_cluster_index][1]
        # get minimum of crnt_costs
        burst_clusters.loc[burst_index] = burst_index, crnt_cluster_number, min(crnt_costs)
        # update cluster performance
        cluster_performance.loc[crnt_cluster_number, ['alignment counter', 'avr cost']] = [cluster_performance.loc[crnt_cluster_number, 'alignment counter'] + 1,
                                                                                          (cluster_performance.loc[crnt_cluster_number, 'alignment counter'] * cluster_performance.loc[crnt_cluster_number, 'avr cost'] + crnt_costs[crnt_cluster_index]) / (cluster_performance.loc[crnt_cluster_number, 'alignment counter'] + 1)]