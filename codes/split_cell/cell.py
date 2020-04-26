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
from point import Point

# geo_range = {'lat_min': 40.953673, 'lat_max': 41.307945, 'lon_min': -8.735152, 'lon_max': -8.156309}
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# FileHandler
file_handler = logging.FileHandler('./log/split_cell_output.log')
file_handler.setLevel(level=logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Cell:
    def __init__(self, trips, cell_num=1000):
        """return the minx, miny, maxx, maxy
        Args:
            trips (dataset of h5py): 
        """
        # self.min_x # longitude
        # self.min_y # latitude

        # in the format of driver: Point(1,2) : num
        self.driver = dict()
        self.point2traj = dict()
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
            # if skip % 10 == 0:
            #     skip += 1
            #     print(skip, total)
            
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

                if point not in self.driver[taxi]:
                    self.driver[taxi][point] = 0
                # record the visited number of the point for taxi_id.
                self.driver[taxi][point] += 1
                # record the point which trajectory visit.
                self.point2traj[point].add(idxx)
        # assert len(self.labels) == len(trips.keys())
        
    def __str__(self):
        f = f'minx, miny, maxx, maxy, gapx, gapy {self.min_x}, {self.min_y}, {self.max_x}, {self.max_y}, {self.gap_x}, {self.gap_y}'
        return f

    def top_k(self, k=20):
        """find the top k of each driver

        Args:
            k (int): knn
        Returns:
            ans (dict): driver: list((Point, frequency))
        """
        ans = dict()
        # drivers (dict): drivers[0][Point(1,2)] = frequency
        drivers = self.driver
        for key in drivers.keys():
            if key not in ans:
                ans[key] = list()
            pairs = drivers[key] # pairs is dict: Point:frequency
            res = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
            ans[key] = res[:k]
        return ans

    def create_groups(self, topk):
        """create groups in the format [[1,2,3], [3,4,5]], list(list)

        Args:
            topk (dict): driver: list([Point, frequency])

        Returns:
            labels: list()
            groups: list(list)
        """
        labels = self.labels
        groups = list()
        visited = set()
        for driver in topk.keys():
            points = topk[driver]
            # import pdb; pdb.set_trace()
            for pair in points:
                if pair[0] not in visited:
                    visited.add(pair[0])
                    cur_group = list(self.point2traj[pair[0]])
                    groups.append(cur_group)
        # import pdb; pdb.set_trace()
        return labels, groups


def one_process(cell_num, k, lock):
    """process one line

    Args:
        num_cell ([type]): [description]
        k ([type]): [description]
        lock ([type]): [description]

    Returns:
        [type]: [description]
    """
    trajectory_path = root_path / 'data' / 'porto2top100_train.h5'
    embedding_path = root_path / 'data' / 'trj_porto2_top100_train180.h5'
    ft = h5py.File(trajectory_path, 'r')

    cell = Cell(ft['trips'], cell_num)
    print(f'begin add trip {cell_num}, {k}')
    t1 = time.time()
    cell.add_trip(ft['trips'], ft['taxi_ids'])

    # topk : (dict): driver: list([Point, frequency]), list has only k numbers
    topk = cell.top_k(k=k)

    labels, groups = cell.create_groups(topk)
    
    pc, pq, rr, DB, B = evaluate(labels, groups)
    try:
        lock.acquire()
        logger.info(f'cells,{cell_num*cell_num}, top-k, {k}, pc {pc:.6f}, pq {pq:.6f}, rr {rr:.6f}, DB {int(DB)}, B {int(B)}')
        print(pc, pq, rr, DB, B)
    except Exception as e:
        print(e)
    finally:
        lock.release()
        ft.close()



if __name__ == '__main__':
    root_path = Path(os.getcwd())
    os.chdir(os.path.dirname(os.getcwd()))
    print(os.getcwd())
    

    lock = Lock()
    p_obj = []

    for i in range(1, 5): #num cell
        for j in range(1, 10): # top-k
            p = Process(target=one_process, args=(i*50, j*5, lock))
            p_obj.append(p)
        print('Waiting for all subprocesses done...')

    for i in p_obj:
        i.start()
    for i in p_obj:
        i.join()
    print('All subprocesses done.')
    






# def non_repeated_ratio(frequency):
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
