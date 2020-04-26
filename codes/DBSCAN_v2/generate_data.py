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


def generate_data(data_path='./data/geolife_top100.txt', output_path='./data/geolife_top100_for_dbscan.txt'):
    """gererate the data in the format of DBSCAN requirments
    
    Args:
        data_path (str): the data of geolife.
    """
    labels, features = read_data(data_path)
    ids = list(range(len(labels)))
    assert len(labels) == len(features), 'driver id not matches features'
    # features = features[:200]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'{len(features)}\t{len(features[0])}\n')
        for i in range(len(features)):
            f.write(f'{i+1}')
            for j in range(len(features[i])):
                f.write(f'\t{int((features[i][j]+1)*5000)}')
            f.write('\n')



if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)
    output_path='./data/geolife_top100_for_dbscan.txt'
    print(output_path)
    generate_data(output_path=output_path)
