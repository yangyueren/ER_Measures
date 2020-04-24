#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd
import random
import math

class LSHHashTble:
    def __init__(self, num_hash_func, dim, w=5):
        """
        initialize the hash funcs in the LSH table
        :param num_hash_func: int
        :param dim: the dimension of the feature
        """
        self.num_hash_func = num_hash_func
        self.dim = dim

        self.a = np.random.normal(size=(self.num_hash_func, self.dim))
        self.w = w
        self.b = np.random.random(self.num_hash_func) * self.w
        self.buckets = dict()

    def add_data(self, ids, features):
        """
        hash the data to the buckets
        :param ids: id correspond to the features
        :param features: np.array, shape: (N, dim)
        :return:
        """
        assert features.shape[1] == self.dim, 'dim error'
        v = features.dot(self.a.T) # N * num_hash_func
        # v += self.b
        # v /= self.w
        v = np.where(v>0, 1, 0)
        v = np.floor(v).astype(np.int)
        for idx, line in zip(ids, v):
            t = tuple(line)
            if t not in self.buckets:
                self.buckets[t] = set()
            self.buckets[t].add(idx)

    def get_blocks(self):
        """

        :return: return the ids in the same bucket
        """
        ans = list()
        for key in self.buckets.keys():
            ans.append(list(self.buckets[key]))
        return ans
