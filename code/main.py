#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd
import random
import logging
from knn import knn
from multi_probe_lsh import multi_probe_lsh
from evaluate import evaluate
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# FileHandler
# file_handler = logging.FileHandler('./log/output.log')
# file_handler.setLevel(level=logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


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


def frun():
    labels, features = read_data('./data/geolife_top100.txt')
    assert len(labels) == len(features), 'driver id not matches features'
    groups_knn = knn(labels, features, K=10)
    # pc_knn, pq_knn, rr_knn = evaluate(labels, groups_knn)
    # print(pc_knn, pq_knn, rr_knn)
    groups_mpl = multi_probe_lsh(labels, features, K=10)


if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)
    frun()
