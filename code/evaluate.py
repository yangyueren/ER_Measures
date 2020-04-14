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

def cal_DB(labels, group):
    """
    return the pairs set that are the same driver id
    :param labels:
    :param group:
    :return: set : (traj1, traj2)  traj1 < traj2
    """
    ans = set()
    for i in range(len(group)):
        for j in range(i):
            if labels[i] == labels[j]:
                assert j < i, 'pair error'
                a = group[i]
                b = group[j]
                m = min(a,b)
                n = max(a,b)
                assert m < n, 'pair error'
                ans.add((m, n))
    return ans

def pc(labels, groups):
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


    db_set = set()
    for group in groups:
        label = labels[group]
        cur_set = cal_DB(label, group)
        db_set = db_set | cur_set
    DB = len(db_set)

    return DB / DE


def pq(labels, groups):
    """Pairs Quality (PQ) corresponds to precision,
    as it estimates the portion of non-redundant comparisons
    that involve matching entities.

    :param ids: np.array, (num,)
    :param labels: np.array (num,)
    :param groups: list(list), one list (ids) represent one group
    :return: PQ(B)=|D(B)|/||B||.
    """

    B = 0 #不需要去重

    db_set = set()
    for group in groups:
        length = len(group)
        B += length*(length-1)/2

        label = labels[group]
        cur_set = cal_DB(label, group)
        db_set = db_set | cur_set
    DB = len(db_set)
    return DB / B

def rr(labels, groups):
    """
    Reduction Ratio (RR) estimates the portion of comparisons
    that are avoided in B with respect to the naive, brute-force approach
    :param ids: np.array, (num,)
    :param labels: np.array (num,)
    :param groups: list(list), one list (ids) represent one group
    :return: RR(B, E)=1-||B||/||E||
    """
    B = 0
    for group in groups:
        length = len(group)
        B += length * (length - 1) / 2
    num = len(labels)
    E = num * (num-1) / 2
    return 1 - B/E

def evaluate(labels, groups):
    """
    return the three result
    :param labels:
    :param groups:
    :return:
    """
    _pc = pc(labels, groups)
    _pq = pq(labels, groups)
    _rr = rr(labels, groups)
    return _pc, _pq, _rr