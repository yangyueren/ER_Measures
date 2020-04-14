#!~/anaconda3/envs/pytor/bin/python
import numpy as np
import pandas as pd
import faiss

def knn(labels, features, K=10):
    features = features.astype('float32')
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)  # build the index

    index.add(features)  # add vectors to the index

    D, I = index.search(features, K)

    return I
