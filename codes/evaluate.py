#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd


def all_np(arr):
    """
    calculate the nums of each element
    :param arr:
    :return:
    """
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

def cal_DB(labels, groups):
    """
    return the pairs set that are the same driver id
    :param labels:
    :param group:
    :return: DB
    """

    db_dict = dict()
    for group in groups:
        label = labels[group]
        cal_DB_2(label, group, db_dict) # db_dict: 1:set(2,3,4), record all found pairs
    DB = 0
    for k in db_dict.keys():
        DB += len(db_dict[k])

    return DB


def cal_DB_2(labels, group, db):
    """
    return the pairs set that are the same driver id
    :param labels:
    :param group:
    :return: set : (traj1, traj2)  traj1 < traj2
    """

    for i in range(len(group)):
        for j in range(i):
            if labels[i] == labels[j]:
                assert j < i, 'pair error'
                a = group[i]
                b = group[j]
                m = min(a,b)
                n = max(a,b)
                assert m < n, 'pair error'
                if m not in db:
                    db[m] = set()
                db[m].add(n)


def pc(labels, groups, DB):
    """
    Pairs Completeness (PC) assesses the portion of the duplicate entities
    that co-occur at least once in B (also known as recall in other research ﬁelds)
    :param ids: np.array, (num,)
    :param labels: np.array (num,)
    :param groups: list(list), one list (ids) represent one group
    :return:PC(B, E)=|D(B)|/|D(E)|
    """
    elem_fre = all_np(labels)
    DE = 0
    for k,v in elem_fre.items():
        tmp = v * (v-1) / 2
        DE += tmp

    # db_dict = dict()
    # for group in groups:
    #     label = labels[group]
    #     cal_DB_2(label, group, db_dict)
    #
    # DB = 0
    # for k in db_dict.keys():
    #     DB += len(db_dict[k])
        
    return DB / DE


def pq(labels, groups, DB):
    """Pairs Quality (PQ) corresponds to precision,
    as it estimates the portion of non-redundant comparisons
    that involve matching entities.

    :param labels: np.array (num,)
    :param groups: list(list), one list (ids) represent one group
    :return: PQ(B)=|D(B)|/||B||.
    """

    B = 0 #不需要去重
    for group in groups:
        length = len(group)
        B += length*(length-1)/2

    # db_dict = dict()
    # for group in groups:
    #     length = len(group)
    #     B += length*(length-1)/2
    #     label = labels[group]
    #     cal_DB_2(label, group, db_dict)
    # DB = 0
    # for k in db_dict.keys():
    #     DB += len(db_dict[k])

    return DB / B, B

def rr(labels, groups, B):
    """
    Reduction Ratio (RR) estimates the portion of comparisons
    that are avoided in B with respect to the naive, brute-force approach
    :param ids: np.array, (num,)
    :param labels: np.array (num,)
    :param groups: list(list), one list (ids) represent one group
    :return: RR(B, E)=1-||B||/||E||
    """
    # B = 0
    # for group in groups:
    #     length = len(group)
    #     B += length * (length - 1) / 2

    num = len(labels)
    E = num * (num-1) / 2
    return 1 - B/E

def evaluate(labels, groups):
    """
    return the three result
    :param labels: list, the driver of the ids in the groups
    :param groups: list(list)
    :return:
    """
    labels = np.array(labels)
    
    DB = cal_DB(labels, groups)
    _pc = pc(labels, groups, DB)
    _pq, B = pq(labels, groups, DB)
    _rr = rr(labels, groups, B)
    return _pc, _pq, _rr, DB, B
