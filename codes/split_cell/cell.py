import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import pickle as pkl


class Cell:
    self.min_x # longitude
    self.min_y # latitude
    self.max_x
    self.max_y
    self.gap_x
    self.gap_y
    self.cell_num_per_side
    self.driver = dict()

    def __init__(self, trips, cell_num=1000):
        """return the minx, miny, maxx, maxy
        Args:
            trips (dataset of h5py): 
        """
        self.cell_num_per_side = cell_num
        self.min_x, self.min_y, self.max_x, self.max_y = self._area(trips)
        self.gap_x, self.gap_y = self._cell_gap(self.cell_num_per_side)

    def _area(self, trips):
        """return the minx, miny, maxx, maxy

        Args:
            trips (dataset of h5py): 
        """
        minx = 10000
        miny = 10000
        maxx = -10000
        maxy = -10000
        for key in trips.keys():
            trip = trips[key][()]
            minx = min(minx, trip[:,1].min())
            maxx = max(maxx, trip[:,1].max())
            miny = min(miny, trip[:,0].min())
            maxy = max(maxy, trip[:,0].max())

        return minx, miny, maxx, maxy

    def _cell_gap(self, nums):
        """find the gap for the target nums

        Args:
            nums (int, optional): the cell number of one side. Defaults to 1000.
        """
        gap_x = (self.max_x - self.min_x) / nums
        gap_y = (self.max_y - self.min_y) / nums
        return gap_x, gap_y

    def add_trip(self, trips, taxi_ids):
        """add trip to the driver

        Args:
            trips ([type]): [description]
        """
        for key_trip, key_taxi in zip(trips.keys(), taxi_ids.keys()):
            trip = trips[key_trip][()]

            taxi = taxi_ids[key_taxi][()]
            x = np.floor((trip[:,1] - self.min_x) / self.gap_x)
            x = x.astype(np.int)
            y = np.floor((trip[:,0] - self.min_y) / self.gap_y)
            y = y.astype(np.int)
            if taxi not in self.driver:
                self.driver[taxi] = list()
            assert len(x) == len(y)
            for i in range(len(x)):
                self.driver[taxi].append((x[i], y[i]))
            

if __name__ == '__main__':
    root_path = Path(os.getcwd()).parent.resolve().parent.resolve()
    os.chdir(os.path.dirname(os.getcwd()))
    print(os.getcwd())
    trajectory_path = root_path / 'data' / 'porto2top100_train.h5'
    embedding_path = root_path / 'data' / 'trj_porto2_top100_train180.h5'
    f = h5py.File(data_path, 'r')
    cell = Cell(f['trips'], 1000)
    cell.add_trip(f['trips'], f['taxi_ids'])
    with open(root_path/'tmp'/'cell.pkl', 'w') as fc:
        pkl.dump(cell, fc)

    # f['trips']
    # f['timestamps']
    # f['taxi_ids']

    f.close()
