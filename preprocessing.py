from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from radar_scenes.sequence import Sequence

def group_timestamps_by_time(timestamps: List[int], time_ms: int) -> List[int]:
    """ Group timestamps in a sequence by time for later temporal agreggation
    of point clouds.
    
    Args:
        timestamps: List of timestamps.
        time_ms: Time in milliseconds to group by.
    Returns:
        frames: List of timestamps grouped by time.
    """

    # Convert time from milliseconds to microseconds.
    time_us = time_ms * 1000

    # Initialize the pivot and the list of frames.
    pivot = timestamps[0]
    frames = []
    timestamps_in_frame = []

    for idx, timestamp in enumerate(timestamps):
        if timestamp - pivot >= time_us:
            frames.append(timestamps_in_frame)
            timestamps_in_frame = []
            pivot = timestamp
        else:
            timestamps_in_frame.append(timestamp)
        if idx == len(timestamps) - 1:
            frames.append(timestamps_in_frame)

    return frames

def aggregated_point_cloud(frame: List[int], seq: Sequence) -> pd.DataFrame:
    """ From a list of timestamps that compose a frame, aggregate their point
    clouds and convert them into the car coordinate system.
    
    Args:
        frame: List of timestamps that compose a frame.
    Returns:
        point_cloud: Pandas dataframe with the aggregated point cloud in car
            coordinate system.
    """

    # Load the first scene and its odometry.
    first_scene = seq.get_scene(frame[0])
    first_odom = first_scene.odometry_data

    # Load the last scene, its odometry and its camera image.
    last_scene = seq.get_scene(frame[-1])
    last_odom = last_scene.odometry_data
    last_image = last_scene.camera_image_name
    
    # Calculate start and end points.
    x0_tr = first_odom[1] - last_odom[1]
    y0_tr = first_odom[2] - last_odom[2]
    x0_rot1 = x0_tr * np.cos(last_odom[3]) - y0_tr * np.sin(last_odom[3])
    y0_rot1 = x0_tr * np.sin(last_odom[3]) + y0_tr * np.cos(last_odom[3])
    x1, y1 = 0, 0

    # Load the point clouds and aggregate them.
    point_cloud = pd.DataFrame(first_scene.radar_data)
    for timestamp in frame[1:]:
        scene = seq.get_scene(timestamp)
        point_cloud = pd.concat([point_cloud, pd.DataFrame(scene.radar_data)], ignore_index=True)

    # Translate the point cloud from the map coordinate system to the car
    # coordinate system.
    point_cloud['x_tr'] = point_cloud['x_seq'] - last_odom[1]
    point_cloud['y_tr'] = point_cloud['y_seq'] - last_odom[2]

    # Rotate the point cloud from the map coordinate system to the car
    # coordinate system.
    yaw = last_odom[3] 
    point_cloud['x_aggr'] = point_cloud['x_tr'] * np.cos(last_odom[3]) - point_cloud['y_tr'] * np.sin(last_odom[3])
    point_cloud['y_aggr'] = point_cloud['x_tr'] * np.sin(last_odom[3]) + point_cloud['y_tr'] * np.cos(last_odom[3])

    # Compensate the yaw.
    inc_x = x1 - x0_rot1
    inc_y = y1 - y0_rot1
    yaw_compensation = np.arctan2(inc_y, inc_x)
    point_cloud['x_mod'] = point_cloud['x_aggr'] * np.cos(np.pi/2-yaw_compensation) - point_cloud['y_aggr'] * np.sin(np.pi/2-yaw_compensation)
    point_cloud['y_mod'] = point_cloud['x_aggr'] * np.sin(np.pi/2-yaw_compensation) + point_cloud['y_aggr'] * np.cos(np.pi/2-yaw_compensation)

    # Rotate x0 and y0 to the car coordinate system.
    x0 = x0_rot1 * np.cos(np.pi/2-yaw_compensation) - y0_rot1 * np.sin(np.pi/2-yaw_compensation)
    y0 = x0_rot1 * np.sin(np.pi/2-yaw_compensation) + y0_rot1 * np.cos(np.pi/2-yaw_compensation)

    return (point_cloud, last_image, x0, y0, x1, y1)

def aggregated_camera_image(frame: List[int], seq: Sequence):
    last_scene = seq.get_scene(frame[-1])
    return last_scene.camera_image_name


def plot_aggregated_point_cloud(point_cloud: pd.DataFrame, x0: float, y0: float, x1: float, y1: float):
    """ Plot the aggregated point cloud.
    """

    # Plot figure
    plt.figure(figsize=(15, 15))
    plt.title('Aggregated point cloud', fontsize=20, fontweight='bold')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')

    # Plot the point cloud.
    plt.scatter(
        point_cloud['x_mod'],
        point_cloud['y_mod'],
        c=point_cloud['vr_compensated'],
        s=1.5,
        cmap='hsv',
    )
    plt.xlim(-50, 50)
    plt.ylim(-25, 75)

    # Plot the initial point.
    plt.vlines(x0, y0 - 1.5, y0 + 1.5, color='r', linestyles='dashed')
    plt.hlines(y0, x0 - 1.5, x0 + 1.5, color='r', linestyles='dashed')
    plt.text(x0 + 0.5, y0 + 0.5, 'Start point', color='r')

    # Plot the final point.
    plt.vlines(x1, y1 - 1.5, y1 + 1.5, color='b', linestyles='dashed')
    plt.hlines(y1, x1 - 1.5, x1 + 1.5, color='b', linestyles='dashed')
    plt.text(x1 + 0.5, y1 + 0.5, 'End point', color='b')

    plt.show()
