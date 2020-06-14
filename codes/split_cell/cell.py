import os
import math
import random
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import pickle as pkl
import time
from tqdm import tqdm
from multiprocessing import Process, Lock
import sys
import os
from pathlib import Path
import logging
from codes.evaluate import evaluate
from .point import Point


class Cell:
    def __init__(self, cell_num, geo_range):
        """init the cell object

        Args:
            cell_num (int): the num of per side
            geo_range (dict): the geo range of the city
        """

        # in the format of driver: Point(1,2) : num
        # self.driver is of no use, just ignore it.
        # self.driver[taxi][point] = frequency
        self.driver = dict()
        # self.point2freq[point] = freq
        self.point2freq = dict()
        # self.point2traj[point].add(idxx)
        self.point2traj = dict()

        # self.point2stop_point_freq[point] = freq
        self.point2stop_point_freq = dict()

        # self.labels[id] is the trajectoyr id 's driver.
        self.labels = list()
        self.cell_num_per_side = cell_num
        self.cell_rank = None
        self.min_x, self.min_y, self.max_x, self.max_y = \
            geo_range['lon_min'], geo_range['lat_min'], geo_range['lon_max'], geo_range['lat_max']

        self.gap_x, self.gap_y = self._cell_gap(self.cell_num_per_side)

    def _cell_gap(self, nums):
        """find the gap for the target nums

        Args:
            nums (int, optional): the cell number of one side. Defaults to 1000.
        """
        gap_x = (self.max_x - self.min_x) / nums
        gap_y = (self.max_y - self.min_y) / nums
        return gap_x, gap_y

    def add_trip(self, df):
        """add trip to the driver

        Args:
            trips ([type]): [description]
        """

        skip = 0
        total = 0
        self.labels = list()
        for idx, traj in tqdm(df.iterrows()):
            trip, taxi, timestamps = np.array(traj['POLYLINE']), int(traj['LABEL']), np.array(traj['TIMESTAMPS'])
            s_point = np.array(traj['s_point']) # [[1.37142289, 103.87005064]]
            self.labels.append(taxi)
            if sum(trip[:, 0] < self.min_x) > 0 or sum(trip[:, 0] > self.max_x) > 0:
                skip += 1
                continue
            if sum(trip[:, 1] < self.min_y) > 0 or sum(trip[:, 1] > self.max_y) > 0:
                skip += 1
                continue
            assert len(trip) == len(timestamps), 'timestamps not match polyline'
            
            x = np.floor((trip[:, 0] - self.min_x) / self.gap_x)
            x = x.astype(np.int)
            y = np.floor((trip[:, 1] - self.min_y) / self.gap_y)
            y = y.astype(np.int)
            if taxi not in self.driver:
                self.driver[taxi] = dict()
            
            assert len(x) == len(y)
            vi = dict()
            duration = timestamps[1:] - timestamps[:-1]
            duration = np.concatenate((duration, [duration[-1]]), axis=0)
            for i in range(len(x)):
                point = Point(x[i], y[i])
                if point not in vi:
                    vi[point] = 0

                vi[point] += 1
                # if vi[point] < len(x) / self.cell_num_per_side:
                #     continue

                if point not in self.point2traj:
                    self.point2traj[point] = set()

                if point not in self.point2freq:
                    self.point2freq[point] = 0

                if point not in self.driver[taxi]:
                    self.driver[taxi][point] = 0

                if point not in self.point2stop_point_freq:
                    self.point2stop_point_freq[point] = 0

                # record the visited number of the point for taxi_id.
                self.driver[taxi][point] += 1
                # record the point which trajectory visit.
                self.point2traj[point].add(idx)
                # record the point visiting frequency.
                self.point2freq[point] += 1
                # if use duration[i], this means record the visiting time.
                # self.point2freq[point] += duration[i]
        
            # import pdb; pdb.set_trace()
            if len(s_point) > 0:
                x_sp = np.floor((s_point[:, 1] - self.min_x) / self.gap_x)
                x_sp = x_sp.astype(np.int)
                y_sp = np.floor((s_point[:, 0] - self.min_y) / self.gap_y)
                y_sp = y_sp.astype(np.int)
                assert len(x_sp) == len(y_sp)
                for i in range(len(x_sp)):
                    point = Point(x_sp[i], y_sp[i])
                    if point not in self.point2stop_point_freq:
                        self.point2stop_point_freq[point] = 0
                    # record the stop point visiting frequency.
                    self.point2stop_point_freq[point] += 1

        print(f'skip={skip}')
        
    def top_k(self, k=20):
        """find the top k cell from all cells.

        Args:
            k (int): top-k cells
        Returns:
            ans (list): list((Point, frequency))
        """
        def is_neighbor(topk, now):
            point2 = now[0]
            l2 = len(self.point2traj[point2])
            for ii in range(len(topk)):
                point1 = topk[ii][0]                
                if abs(point1.x - point2.x) == 0 and abs(point1.y - point2.y) <= 1:
                    return True
                if abs(point1.x - point2.x) <= 1 and abs(point1.y - point2.y) == 0:
                    return True

                if len(self.point2traj[point1].intersection(self.point2traj[point2])) > l2*0.7:
                    return True

            return False

        ans = list()
        # import pdb; pdb.set_trace()
        # res = sorted(self.point2freq.items(), key=lambda x: math.sqrt(x[1])* (1.0 / (len(self.point2traj[x[0]]) + 1)), reverse=True)
        # res = sorted(self.point2freq.items(), key=lambda x: (x[1])* (1.0 / (len(self.point2traj[x[0]]) + 1)), reverse=True)
        res = sorted(self.point2freq.items(), key=lambda x: (x[1] * math.sqrt(1+self.point2stop_point_freq[x[0]]))* (1.0 / (np.power(len(self.point2traj[x[0]]), ) + 1)), reverse=True)
        self.cell_rank = res
        i = 0
        j = 0
        while i<k and j < len(res):
            tmp = res[j]
            j += 1
            if len(self.point2traj[tmp[0]]) > len(self.labels) / 4:
                continue
            if len(self.point2traj[tmp[0]]) < 4:
                continue

            if not is_neighbor(ans, tmp) or random.random() < 0.1:
                ans.append(tmp)
                i += 1
        return ans
    

    def create_groups(self, topk):
        """create groups in the format [[1,2,3], [3,4,5]], list(list)
            where elements are trajectory id.
            labels[i] stores the trajectory_i's driver_id

        Args:
            topk (list): list([Point, frequency])

        Returns:
            labels: list()
            groups: list(list)
        """
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

    def show_info(self, topk):
        """log the cell info.

        Args:
            topk (list): list([Point, frequency])
        """
        po = [i[0] for i in topk]
        with open('./log/info.log', 'a+', encoding='utf-8') as f:
            f.write(f'begin: {self.cell_num_per_side}\n')
            for i in range(len(self.cell_rank)):
                k = self.cell_rank[i][0]
                if k in po:
                    f.write('choose: ')
                f.write(f'{k},rank {i}, {len(self.point2traj[k])}, {self.point2freq[k]}, {self.point2stop_point_freq[k]}\n')

    def __str__(self):
        f = f'minx, miny, maxx, maxy, gapx, gapy {self.min_x}, {self.min_y}, {self.max_x}, {self.max_y}, {self.gap_x}, {self.gap_y}'
        return f

    def __repr__(self):
        f = f'minx, miny, maxx, maxy, gapx, gapy {self.min_x}, {self.min_y}, {self.max_x}, {self.max_y}, {self.gap_x}, {self.gap_y}'
        return f