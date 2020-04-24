#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd
import random
import logging
from knn import knn
from multi_probe_lsh import multi_probe_lsh
from evaluate import evaluate
import lsh_p_stable
import lsh_random_hyperplane
import lsh_simhash

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# FileHandler
file_handler = logging.FileHandler('./log/output.log')
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


def frun(K=10):
    """knn blocking
    
    Args:
        K (int, optional): nearest neighbors. Defaults to 10.
    """
    labels, features = read_data('./data/geolife_top100.txt')
    assert len(labels) == len(features), 'driver id not matches features'
    groups_knn = knn(labels, features, K=K)
    pc_knn, pq_knn, rr_knn = evaluate(labels, groups_knn)
    logger.info(f',pc_knn,{pc_knn},pq_knn,{pq_knn},rr_knn,{rr_knn}')

    groups_mpl = multi_probe_lsh(labels, features, K=K)
    pc_mpl, pq_mpl, rr_mpl = evaluate(labels, groups_mpl)
    logger.info(f',pc_mpl,{pc_mpl},pq_mpl,{pq_mpl},rr_mpl,{rr_mpl}')


def p_stable_run(num_ht = 4, num_hf = 8, w=30):
    """lsh p stable
    
    Args:
        num_ht (int, optional): hash tables. Defaults to 4.
        num_hf (int, optional): hash functions for each hash table. Defaults to 8.
        w (int, optional): hyperparameter. Defaults to 30.
    """
    labels, features = read_data('./data/geolife_top100.txt')
    ids = list(range(len(labels)))
    assert len(labels) == len(features), 'driver id not matches features'

    lsh_index = lsh_p_stable.LSH(num_ht, num_hf, features.shape[1], w)
    lsh_index.add_data(ids, features)
    groups_lsh = lsh_index.get_blocks()
    pc_lsh, pq_lsh, rr_lsh, DB, B = evaluate(labels, groups_lsh)

    logger.info(f'p_stable - w {w:2d}, {num_ht:2d} hash tables, {num_hf:2d} hash functions, pc {pc_lsh:.6f}, pq {pq_lsh:.6f}, rr {rr_lsh:.6f}, DB {int(DB)}, B {int(B)}')


if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)
    for w in range(2, 5):
        for i in range(1,11):
            for j in range(1,11):
                try:
                    p_stable_run(i, j, w*5+20)
                    print(f'{w} {i} hash tables {j} hash funcs is done.')
                except Exception as e:
                    print(e)
    # for i in range(1, 2):
    #     try:
    #         frun(i*10)
    #         print(f'{i*10} is done.')
    #     except Exception as e:
    #         print(e)
