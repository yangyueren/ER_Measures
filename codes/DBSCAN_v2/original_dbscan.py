#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd
import random
import logging
import sys
sys.path.append(".")
import os
from pathlib import Path
from multiprocessing import Process, Lock
from sklearn.cluster import DBSCAN
from codes.evaluate import evaluate

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# FileHandler
file_handler = logging.FileHandler('./log/dbscan_original_output.log')
file_handler.setLevel(level=logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def read_data(file_name='./data/geolife_top100.txt'):
    all = list()
    with open(file_name, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f.readlines():
            li = line.replace(' ', ',').split(',')
            all.append(li)

    all = np.array(all).astype(np.float)
    labels = np.array(all[:,0]).astype(np.int)
    features = np.array(all[:,1:]).astype(np.float32)
    return labels, features



def calculate_score(eps, minPts, lock, data_path='./data/geolife_top100.txt'):
    """evaluate the result

    Args:
        logfile (str): store the score
        data_path (str, optional): [description]. Defaults to './data/geolife_top100.txt'.
        result_path (str, optional): [description]. Defaults to './data/2D_Visual_minPts=20_eps=5000.0_OurApprox'.
    """
    labels, features = read_data(data_path)
    ids = list(range(len(labels)))
    assert len(labels) == len(features), 'driver id not matches features'
    clustering = DBSCAN(eps=eps, min_samples=minPts).fit(features)
    cluster = clustering.labels_
    keys = np.unique(cluster)
    groups = list()
    num = list()
    for key in keys:
        index = np.where(cluster == key)[0]
        group = index.tolist()
        groups.append(group)
        num.append(len(group))
    # import pdb; pdb.set_trace()
    
    
    print('begin evaluate:')
    pc, pq, rr, DB, B = evaluate(labels, groups)
    try:
        lock.acquire()
        logger.info(f'eps {eps:.5f}, minPts {minPts:3d}, pc {pc:.6f}, pq {pq:.6f}, rr {rr:.6f}, DB {int(DB)}, B {int(B)}')
        logger.info(f'{str(num)}')
    except Exception as e:
        print(e)
    finally:
        lock.release()
        print(pc, pq, rr, DB, B)


if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)
    # generate_data()
    root_path = Path(os.getcwd())
    
    lock = Lock()
    p_obj = []
    try:
        for i in range(1, 6):
            for j in range(1, 10):
                eps = pow(10, i) * 0.1
                minPts = j * 5
                # calculate_score(eps, minPts)
                p = Process(target=calculate_score, args=(eps, minPts, lock))
                p_obj.append(p)
        print('Waiting for all subprocesses done...')

        for i in p_obj:
            i.start()
        for i in p_obj:
            i.join()
        print('All subprocesses done.')
    except Exception as e:
        print(e)
    





