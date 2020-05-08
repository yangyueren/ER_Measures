import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import pickle as pkl
import time
from tqdm import tqdm
from multiprocessing import Process, Lock
import sys
sys.path.append(".")
import os
from pathlib import Path
import logging
from codes.evaluate import evaluate
sys.path.append('./codes/split_cell/')
from point import Point

# geo_range = {'lat_min': 40.953673, 'lat_max': 41.307945, 'lon_min': -8.735152, 'lon_max': -8.156309}

class Cell:
    def __init__(self, trips, cell_num=1000):
        """return the minx, miny, maxx, maxy
        Args:
            trips (dataset of h5py): 
        """
        # self.min_x # longitude
        # self.min_y # latitude

        # in the format of driver: Point(1,2) : num
        # self.driver[taxi][point] = frequency
        self.driver = dict()
        # self.point2freq[point] = freq
        self.point2freq = dict()
        # self.point2traj[point].add(idxx)
        self.point2traj = dict()
        

        # self.labels[id] is the trajectoyr id 's driver.
        self.labels = list()
        self.cell_num_per_side = cell_num
        self.min_x, self.min_y, self.max_x, self.max_y = -8.735152, 40.953673, -8.156309, 41.307945
        # self.min_x, self.min_y, self.max_x, self.max_y = self._area(trips)
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
            # print(key)
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
        # lati : 40, trip[:, 1] y
        # longti: -8, trip[:, 0] x
        skip = 1
        total = 1
        self.labels = list()
        # trj_num = len(trips.keys())
        trj_num = 30000
        # for idx in tqdm(range(len(trips.keys()) )):
        for idxx in range(trj_num):
            idx = idxx * 12
            total += 1
            # idx is id of trajectory, its driver is in taix_ids[str(idx)]
            trip = trips[str(idx)][()]
            taxi = int(taxi_ids[str(idx)][()])
            # record the driver id for the trajectory idx
            self.labels.append(taxi)

            if sum(trip[:, 0] < self.min_x) > 0 or sum(trip[:, 0] > self.max_x) > 0:
                skip += 1
                continue
            if sum(trip[:, 1] < self.min_y) > 0 or sum(trip[:, 1] > self.max_y) > 0:
                skip += 1
                continue
            
            x = np.floor((trip[:, 0] - self.min_x) / self.gap_x)
            x = x.astype(np.int)
            y = np.floor((trip[:, 1] - self.min_y) / self.gap_y)
            y = y.astype(np.int)
            if taxi not in self.driver:
                self.driver[taxi] = dict()
            
            assert len(x) == len(y)
            for i in range(len(x)):
                point = Point(x[i], y[i])
                if point not in self.point2traj:
                    self.point2traj[point] = set()

                if point not in self.point2freq:
                    self.point2freq[point] = 0

                if point not in self.driver[taxi]:
                    self.driver[taxi][point] = 0

                # record the visited number of the point for taxi_id.
                self.driver[taxi][point] += 1
                # record the point which trajectory visit.
                self.point2traj[point].add(idxx)
                # record the point visiting frequency.
                self.point2freq[point] += 1
        # assert len(self.labels) == len(trips.keys())
        
    def __str__(self):
        f = f'minx, miny, maxx, maxy, gapx, gapy {self.min_x}, {self.min_y}, {self.max_x}, {self.max_y}, {self.gap_x}, {self.gap_y}'
        return f
    def __repr__(self):
        f = f'minx, miny, maxx, maxy, gapx, gapy {self.min_x}, {self.min_y}, {self.max_x}, {self.max_y}, {self.gap_x}, {self.gap_y}'
        return f


    def top_k(self, k=20):
        """find the top k of each driver

        Args:
            k (int): knn
        Returns:
            ans (dict): driver: list((Point, frequency))
        """
        def is_neighbor(topk, now):
            for ii in range(len(topk)):
                point1 = topk[ii][0]
                point2 = now[0]
                if abs(point1.x - point2.x) == 0 and abs(point1.y - point2.y) <=1:
                    return True
                if abs(point1.x - point2.x) == 1 and abs(point1.y - point2.y) ==0:
                    return True
            return False

        ans = list()
        # import pdb; pdb.set_trace()
        # res = sorted(self.point2freq.items(), key=lambda x: np.log(x[1])* (1.0 / (len(self.point2traj[x[0]]) + 1)), reverse=True)
        # res = sorted(self.point2freq.items(), key=lambda x: x[1]* (1.0 / (len(self.point2traj[x[0]]) + 1)), reverse=True)
        res = sorted(self.point2freq.items(), key=lambda x: x[1]* (1.0 / (len(self.point2traj[x[0]])*len(self.point2traj[x[0]]) + 1)), reverse=True)
        i = 0
        j = 0
        while i<k and j < len(res):
            tmp = res[j]
            j += 1
            if not is_neighbor(ans, tmp):
                ans.append(tmp)
                i += 1
        
        return ans
    

    def create_groups(self, topk):
        """create groups in the format [[1,2,3], [3,4,5]], list(list)

        Args:
            topk (list): list([Point, frequency])

        Returns:
            labels: list()
            groups: list(list)
        """
        # def is_neighbor(topk):
        #     new_topk = set()
        #     for i in range(len(topk)):
        #         flag = True
        #         for pair in new_topk:
        #             point1 = topk[i][0]
        #             point2 = pair[0]
        #             if abs(point1.x - point2.x) <= 1 and abs(point1.y - point2.y) <=1:
        #                 flag = False
        #                 break
        #         if flag:
        #             new_topk.add(topk[i])
        #     return new_topk

        labels = self.labels
        groups = list()
        visited = set()
        # topk = is_neighbor(topk)
        for pair in topk:
            point = pair[0]
            if point not in visited:
                visited.add(point)
                tmp = list(self.point2traj[point])
            if len(tmp) > 2:
                groups.append(tmp)
        # import pdb; pdb.set_trace()
        # print(groups)
        return labels, groups


    # def top_k(self, k=20):
    #     """find the top k of each driver

    #     Args:
    #         k (int): knn
    #     Returns:
    #         ans (dict): driver: list((Point, frequency))
    #     """
    #     ans = dict()
    #     # drivers (dict): drivers[0][Point(1,2)] = frequency
    #     drivers = self.driver
    #     for key in drivers.keys():
    #         if key not in ans:
    #             ans[key] = list()
    #         pairs = drivers[key] # pairs is dict: Point:frequency
    #         # res = sorted(pairs.items(), key=lambda x: x[1]* 1.0 / self.point2freq[x[0]], reverse=True)
    #         res = sorted(pairs.items(), key=lambda x: x[1]* (1.0 / (len(self.point2traj[x[0]]) + 1)), reverse=True)

    #         ans[key] = res[:k]
    #     return ans
    

    # def neighbor_ratio(self, topk):
    #     """calculate the neighbor ratio for each driver

    #     Args:
    #         topk (dict): driver: list([Point, frequency])
        
    #     Returns:
    #         neighbor_ratio (list): len(driver)+1, the 1st is overall ratio.
    #     """
    #     def is_neighbor(point1, point2):
    #         if abs(point1.x - point2.x) <= 1 and abs(point1.y - point2.y) <=1:
    #             return True
    #         else:
    #             return False

    #     ratio_list = list()
    #     for driver in topk.keys():
    #         buckets = topk[driver]
    #         neighbor = set()
    #         for i in range(len(buckets)-1):
    #             for j in range(i+1, len(buckets)):
    #                 if is_neighbor(buckets[i][0], buckets[j][0]):
    #                     neighbor.add(buckets[i][0])
    #                     neighbor.add(buckets[j][0])
    #         cur_ratio = len(neighbor) / len(buckets)
    #         ratio_list.append(cur_ratio)
    #     average_ratio = sum(ratio_list) / len(ratio_list)
    #     ratio_list.insert(0, average_ratio)
    #     return ratio_list
                        



    # def create_groups(self, topk):
    #     """create groups in the format [[1,2,3], [3,4,5]], list(list)

    #     Args:
    #         topk (dict): driver: list([Point, frequency])

    #     Returns:
    #         labels: list()
    #         groups: list(list)
    #     """
    #     labels = self.labels
    #     groups = list()
    #     visited = set()
    #     for driver in topk.keys():
    #         points = topk[driver]
    #         # import pdb; pdb.set_trace()
    #         cur_group = list()
    #         for pair in points:
    #             if pair[0] not in visited:
    #                 visited.add(pair[0])
    #                 tmp = list(self.point2traj[pair[0]])
    #                 cur_group.extend(tmp)
    #         groups.append(list(set(cur_group)))
    #     # import pdb; pdb.set_trace()
    #     return labels, groups



    # def non_repeated_ratio(self, frequency):
    #     """analyze the non repeated ratio of top k cells for each driver

    #     Args:
    #         frequency (dict): driver: list, list is [(Point, fre)]
    #     Return:
    #         ratio (dict): non repeated ratio for each driver
    #     """
    #     ratio = dict()
    #     for key in frequency:
    #         gt_set = set([i[0] for i in frequency[key]])
    #         junk_set = set()
    #         for jk in frequency:
    #             if jk != key:
    #                 cur_set = set([i[0] for i in frequency[jk]])
    #                 junk_set |= cur_set
    #         cur_ratio = len(gt_set.intersection(junk_set)) / len(gt_set)
    #         ratio[key] = cur_ratio
    #     return ratio
