import os
import sys
import numpy as np
import pandas as pd
import random
from pathlib import Path
import pickle as pkl

from .extract_feature import DrivingState

def read_data(file):
    """read data from file

    Args:
        file (file path): 
    """
    with open(file, 'rb') as f:
        df = pkl.load(f)
    coords_list, driverids_list = df['POLYLINE'].tolist(), df['TAXI_ID'].tolist()
    timestamps = df['TIMESTAMP'].tolist()
    timestamps_list = []
    for cur_time, coords in zip(timestamps, coords_list):
        length = len(coords)
        cur_time_list = list(range(cur_time, cur_time+15*length, 15))
        timestamps_list.append(cur_time_list)

    assert len(coords_list)  == len(timestamps_list), 'error'
    assert len(coords_list)  == len(driverids_list), 'error'
    return np.array(coords_list), np.array(timestamps_list), np.array(driverids_list)

def store_data(data, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

if __name__ == '__main__':
    print(os.getcwd())
    random.seed(10086)
    np.random.seed(10086)
    root_path = Path(os.getcwd())
    data_path = 'porto2_sample10k.pkl'
    abs_data_path = root_path / 'data' / data_path
    coords_list, timestamps_list, driverids_list = read_data(abs_data_path)
    # import pdb; pdb.set_trace()
    seq_list = []
    graph_list = []
    for coords, timestamps, driver_id in zip(coords_list, timestamps_list, driverids_list):
        trajectory = DrivingState(coords, timestamps, driver_id)
        # import pdb; pdb.set_trace()
        seq_vec = trajectory.get_seq_vector()
        graph_vec = trajectory.get_graph_vector()
        seq_list.append(seq_vec)
        graph_list.append(graph_vec)

    df = pd.DataFrame({'SEQ_VEC': seq_list, 'GRAPH_VEC': graph_list})
    assert len(seq_list) == len(graph_list), 'error'
    # assert len(seq_list) == len(coords_list), 'error'
    ans_path = root_path / 'data' / 'porto2_sample10k_transition_feature.pkl'
    store_data(df, ans_path)

        



