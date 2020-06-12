import os
import random
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import pickle as pkl
import time
from tqdm import tqdm
from multiprocessing import Process, Lock
import sys
# sys.path.append(".")
import os
from pathlib import Path
import logging
from codes.evaluate_no_duplicate import evaluate
from .point import Point
from .cell import Cell
from .config import setting

# geo_range = {'lat_min': 40.953673, 'lat_max': 41.307945, 'lon_min': -8.735152, 'lon_max': -8.156309}
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# FileHandler
file_handler = logging.FileHandler('./log/split_cell.log')
file_handler.setLevel(level=logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def one_process(cell_num, k, lock):
    """process one line

    Args:
        num_cell ([type]): [description]
        k ([type]): [description]
        lock ([type]): [description]

    Returns:
        [type]: [description]
    """
    root_path = Path(os.getcwd())
    test_pair_path = setting['pair_path'] + 'test.csv'
    test_pair = pd.read_csv(test_pair_path, usecols=[0, 1, 2])
    test_idx = list(set(test_pair['ltable_id']).union(set(test_pair['rtable_id'])))
    trajectory_path = setting['raw_data_path']
    with open(trajectory_path, 'rb') as f:
        df = pkl.load(f)
        if 'TIMESTAMPS' not in df.columns:
            df.rename(columns={'TIMESTAMP':'TIMESTAMPS'}, inplace=True)
    # df = df.reindex(test_idx)
    df.index = list(range(len(df)))

    geo_range = setting['geo_range']
    cell = Cell(cell_num, geo_range)
    print(f'begin add trip {cell_num}, {k}')
    t1 = time.time()
    cell.add_trip(df)

    # topk : (list): list([Point, frequency]), list has only k numbers
    topk = cell.top_k(k=k)
    # import pdb; pdb.set_trace()
    # groups: list(list), every element is an trajectory id.
    # labels[id] represents the driver of trajectory id.
    labels, groups = cell.create_groups(topk)
    k = len(groups)
    # import pdb; pdb.set_trace()
    
    
    pc, pq, rr, DB, B = evaluate(labels, groups)
    try:
        lock.acquire()
        # logger.info(f'{str(neighbor_ratio)}')
        logger.info(f'cells,{cell_num*cell_num}, top-k, {k}, pc {pc:.6f}, pq {pq:.6f}, rr {rr:.6f}, DB {int(DB)}, B {int(B)}')
        print(pc, pq, rr, DB, B)
    except Exception as e:
        print(e)
    finally:
        lock.release()
        


if __name__ == '__main__':
    
    os.chdir('/home/yangyueren/code/ER_Measures')
    print(os.getcwd())
    random.seed(10086)
    
    lock = Lock()
    p_obj = []
    # one_process(16,7, lock)
    for i in range(3, 5): #num cell
        for j in range(1, 5): # top-k
            p = Process(target=one_process, args=(i*2, j*3, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    for i in range(3, 5): #num cell
        for j in range(3, 5): # top-k
            p = Process(target=one_process, args=(i*5, j*8, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    for i in range(5, 8): #num cell
        for j in range(1, 3): # top-k
            p = Process(target=one_process, args=(i*2, j*10, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    # one_process(50,50, lock)
    for i in range(5, 8): #num cell
        for j in range(5, 7): # top-k
            p = Process(target=one_process, args=(i*3, j*10, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()


    print('All subprocesses done.')
    