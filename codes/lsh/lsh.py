#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd
import random
import math
from .lsh_ht import LSHHashTble

class LSH:
    def __init__(self, num_ht, num_hf, dim):
        """
        initialize the parameters
        :param num_ht: number of hash tables
        :param num_hf: number of hash functions in each hash tables
        :param dim: the dimension of the feature
        """
        self.num_ht = num_ht
        self.num_hf = num_hf
        self.dim = dim
        self.hts = list()
        for i in range(self.num_ht):
            ht = LSHHashTble(self.num_hf, self.dim)
            self.hts.append(ht)

    def add_data(self, ids, features):
        """
        hash the data to the buckets
        :param ids: id correspond to the features
        :param features: np.array, shape: (N, dim)
        :return:
        """
        for ht in self.hts:
            ht.add_data(ids, features)

    def get_blocks(self):
        """
        get the blocks of the hts
        :return:
        """
        ans = list()
        for ht in self.hts:
            tmp = ht.get_blocks() # two dimension
            ans += tmp
        return ans