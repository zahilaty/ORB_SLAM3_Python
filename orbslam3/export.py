"""Generic result export; independent of core tracking and mapping."""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import inverse_pose


def save_trajectory(path, trajectory):
    """CSV: seconds,map_id,state,world camera translation,xyzw quaternion."""
    rows = []
    for timestamp, pose_cw, map_id, state in trajectory:
        pose_wc = inverse_pose(pose_cw)
        quaternion = Rotation.from_matrix(pose_wc[:3, :3]).as_quat()
        rows.append([timestamp, map_id, int(state), *pose_wc[:3, 3], *quaternion])
    values = np.asarray(rows).reshape(-1, 10)
    np.savetxt(Path(path), values, delimiter=",", header="timestamp,map_id,state,x,y,z,qx,qy,qz,qw", comments="")


def save_points(path, atlas):
    rows = [[map_.id, point.id, *point.position] for map_ in atlas.maps.values()
            for point in map_.points.values() if not point.bad]
    np.savetxt(Path(path), np.asarray(rows).reshape(-1, 5), delimiter=",", header="map_id,point_id,x,y,z", comments="")
