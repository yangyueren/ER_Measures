#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd
import random
import falconn

def multi_probe_lsh(labels, features, K=10):
    features = features.astype('float32')
    features -= np.mean(features, axis=0)
    num_points, dim = features.shape
    parms = falconn.get_default_parameters(num_points, dim)
    falconn.compute_number_of_hash_functions(7, parms)

    lsh_index = falconn.LSHIndex(parms)
    lsh_index.setup(features)

    query = lsh_index.construct_query_object(K)
    I = list()
    for i in range(len(features)):
        ans = query.find_k_nearest_neighbors(features[i], K)
        I.append(ans)
    return I
    # import pdb; pdb.set_trace()