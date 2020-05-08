import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import pickle as pkl
import time
from tqdm import tqdm
from multiprocessing import Process, Lock
import sys
sys.path.append(".")
import os
from pathlib import Path
import logging
from codes.evaluate import evaluate
from point import Point
from cell import Cell

# geo_range = {'lat_min': 40.953673, 'lat_max': 41.307945, 'lon_min': -8.735152, 'lon_max': -8.156309}
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# FileHandler
file_handler = logging.FileHandler('./log/debug_topk_3w2multiply_no_duplicate_split_cell_output.log')
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
    trajectory_path = root_path / 'data' / 'porto2top100_train.h5'
    embedding_path = root_path / 'data' / 'trj_porto2_top100_train180.h5'
    ft = h5py.File(trajectory_path, 'r')

    cell = Cell(ft['trips'], cell_num)
    print(f'begin add trip {cell_num}, {k}')
    t1 = time.time()
    cell.add_trip(ft['trips'], ft['taxi_ids'])

    # topk : (dict): driver: list([Point, frequency]), list has only k numbers
    topk = cell.top_k(k=k)
    labels = cell.labels
    point2traj = cell.point2traj

    # neighbor_ratio = cell.neighbor_ratio(topk)
    # print(neighbor_ratio)

    # groups: list(list), every element is an trajectory id.
    # labels[id] represents the driver of trajectory id.
    labels, groups = cell.create_groups(topk)
    
    
    
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
        ft.close()


if __name__ == '__main__':
    
    os.chdir('/home/yangyueren/code/ER_Measures')
    print(os.getcwd())
    
    lock = Lock()
    p_obj = []
    # one_process(50,5, lock)
    for i in range(1, 5): #num cell
        for j in range(1, 20): # top-k
            p = Process(target=one_process, args=(i*10, j*10, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    # one_process(50,50, lock)
    for i in range(1, 10): #num cell
        for j in range(1, 20): # top-k
            p = Process(target=one_process, args=(i*10, j*10, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    # one_process(50,50, lock)
    for i in range(1, 10): #num cell
        for j in range(1, 20): # top-k
            p = Process(target=one_process, args=(i*15, j*10+20, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    # one_process(50,50, lock)
    for i in range(1, 5): #num cell
        for j in range(20, 60): # top-k
            p = Process(target=one_process, args=(i*10, j*3, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    p_obj = []
    # one_process(50,50, lock)
    for i in range(1, 10): #num cell
        for j in range(10, 60): # top-k
            p = Process(target=one_process, args=(i*20, j*20, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()

    print('All subprocesses done.')
    






# def non_repeated_ratio(frequency):
#     """analyze the non repeated ratio of top k cells for each driver

#     Args:
#         frequency (dict): driver: list, list is [(Point, fre)]
#     Return:
#         ratio (dict): non repeated ratio for each driver
#     """
#     ratio = dict()
#     for key in frequency:
#         gt_set = set([i[0] for i in frequency[key]])
#         junk_set = set()
#         for jk in frequency:
#             if jk != key:
#                 cur_set = set([i[0] for i in frequency[jk]])
#                 junk_set |= cur_set
#         cur_ratio = len(gt_set.intersection(junk_set)) / len(gt_set)
#         ratio[key] = cur_ratio
#     return ratio
