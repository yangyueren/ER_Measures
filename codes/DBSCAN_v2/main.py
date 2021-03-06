#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd
import random
import logging
import sys
sys.path.append(".")
import os
from pathlib import Path
from codes.evaluate import evaluate

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# FileHandler
file_handler = logging.FileHandler('./log/dbscan_output.log')
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



def calculate_score(result_path, data_path='./data/geolife_top100.txt'):
    """evaluate the result

    Args:
        logfile (str): store the score
        data_path (str, optional): [description]. Defaults to './data/geolife_top100.txt'.
        result_path (str, optional): [description]. Defaults to './data/2D_Visual_minPts=20_eps=5000.0_OurApprox'.
    """
    labels, features = read_data(data_path)
    ids = list(range(len(labels)))
    assert len(labels) == len(features), 'driver id not matches features'
    groups = list()
    with open(result_path, 'r', encoding='utf-8') as f:
        num = 0
        for num,line in enumerate(f.readlines()):
            if num == 0:
                continue
            if num % 2 == 1:
                continue
            line = line.replace('\n', '')
            line = line.split(' ')
            # minu 1 because id plus 1.
            new_line = [int(int(i)-1) for i in line if i.isnumeric()]
            groups.append(new_line)
    print('begin evaluate:')
    pc, pq, rr, DB, B = evaluate(labels, groups)

    logger.info(f'{str(result_path).split("/")[-1]}, pc {pc:.6f}, pq {pq:.6f}, rr {rr:.6f}, DB {int(DB)}, B {int(B)}')

    print(pc, pq, rr, DB, B)


if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)
    # generate_data()
    root_path = Path(os.getcwd())
    data_path = root_path / 'data' / 'dbscan_data'
    for file in os.listdir(data_path):
        print(file)
        calculate_score(data_path / file)
