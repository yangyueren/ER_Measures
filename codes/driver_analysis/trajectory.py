import numpy as np
import pandas as pd
from geopy.distance import distance
import folium


class Trajectory:
    """Calculate and store the feature of single trajectory"""
    def __init__(self, coords, timestamps, driver_id):
        """
        Args:
        - coords: np.ndarray, [(lon1, lat1), (lon2, lat2), ...]
        - timestamps: np.ndarray, unix time, same length as coords
        - driver_id: np.ndarray,
        """
        self.distances = None
        self.speeds = None
        self.angles = None
        self.accelerations = None
        self.coords = coords
        self.timestamps = timestamps
        self.driver_id = driver_id
        self.get_accleration()
        self.get_angles()


    def get_distance(self):
        if self.distances is not None:
            return self.distances
        prev_coords = self.coords[:-1]
        after_coords = self.coords[1:]
        # geopy.distance.distace accept (lat, lon) input
        distances = [distance((prev[1], prev[0]), (after[1], after[0])).meters for prev, after in zip(prev_coords, after_coords)]
        self.distances = np.array(distances)
        return self.distances

    def get_speed(self):
        if self.speeds is not None:
            return self.speeds
        self.time_diff = self.timestamps[1:] - self.timestamps[:-1]
        distances = self.get_distance()
        speeds = distances / self.time_diff
        speeds = np.concatenate((np.array([0.0]), speeds), axis=0)  # add zero speed for the first point
        normal_idx = speeds < 38.9  # 140km/h, 140 / 3.6 = 38.8888
        if False in normal_idx:  # directly delete the abnormal part
            # print(normal_idx)
            speeds = speeds[normal_idx]
            self.distances = distances[normal_idx[1:]]
            self.time_diff = self.time_diff[normal_idx[1:]]
            self.coords = self.coords[normal_idx]
        self.speeds = speeds
        return self.speeds

    def get_accleration(self):
        if self.accelerations is not None:
            return self.accelerations
        speeds = self.get_speed()
        speeds_diff = speeds[1:] - speeds[:-1]
        self.accelerations = np.concatenate((np.array([0.0]), (speeds_diff / self.time_diff)), axis=0)  # add zero accleration for the first point
        return self.accelerations

    def get_angles(self):
        angles = []
        for idx, coord in enumerate(self.coords[:-1]):
            # angles.append(np.arccos(np.dot(coord,self.coords[idx+1])/(np.linalg.norm(coord)*np.linalg.norm(self.coords[idx+1]))))  # no, arccos is not good for this
            
            if self.distances[idx] >= 0.1:  # it's meaningless to compute the angle when drving real small distance
                coord_diff = self.coords[idx+1] - coord
                angles.append(np.arctan(coord_diff[1] / coord_diff[0]))
            else:
                if len(angles) > 0:
                    angles.append(angles[-1])
                else:
                    angles.append(0.0)
        self.angles = angles + [angles[-1]]  # add for last point
        return self.angles

    def vis(self, with_marker=False):
        """Visualization of this single trajectory
        
        Used in jupyter notebook for debug.
        Attention: folium accept trajectories in (lat, lon) format.
        Args:
        - with_marker: boolean, 
        """
        start_point = self.coords[0]
        traj = [(cd[1], cd[0]) for cd in self.coords]
        m = folium.Map(location=traj[0], zoom_start=13)
        folium.PolyLine(locations=traj, color='blue').add_to(m)
        if with_marker:
            for idx, cd in enumerate(traj):
                tool_tip = "pid:%d\nspeed:%.2fm/s\nacceleration:%.4f\nangles:%.4f" % (idx, self.speeds[idx], self.accelerations[idx], self.angles[idx])
                folium.Marker(location=cd, tooltip=tool_tip).add_to(m)
        return m