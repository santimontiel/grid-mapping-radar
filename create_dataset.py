import os
from rich.progress import track
import numpy as np
import cv2 as cv

from preprocessing import group_timestamps_by_time, aggregated_point_cloud, \
    category_num_map, category_sem_map
from radar_scenes.sequence import Sequence, get_training_sequences, \
    get_validation_sequences

def rcs_scale(value: float) -> float:

    if value >= 10.0:
        return 1.0
    elif value <= -10.0:
        return 0.0
    else:
        return (value + 10.0) / 20.0 

def vr_scale(value: float) -> float:

    if value >= 7.50:
        return 1.0
    elif value <= -7.50:
        return 0.0
    else:
        return (value + 7.5) / 15.0

# Load RadarScenes dataset.
PATH_TO_DATASET = "/home/robesafe/Datasets/RadarScenes"
if not os.path.exists(PATH_TO_DATASET):
    raise FileNotFoundError("Dataset not found at {}".format(PATH_TO_DATASET))

# Set the path to the new dataset.
PATH_TO_NEW_DATASET = "/home/robesafe/Santi/radarscenes_bev_yolo_v2"
if not os.path.exists(PATH_TO_NEW_DATASET):
    os.makedirs(PATH_TO_NEW_DATASET)
os.makedirs(os.path.join(PATH_TO_NEW_DATASET, "images", "train"))
os.makedirs(os.path.join(PATH_TO_NEW_DATASET, "images", "val"))

# Get the training sequences.
training_sequences = get_training_sequences(os.path.join(PATH_TO_DATASET, "data", "sequences.json"))
validation_sequences = get_validation_sequences(os.path.join(PATH_TO_DATASET, "data", "sequences.json"))

# Number of sequences
START_SEQ, END_SEQ = 1, 158

for seq_number in range(START_SEQ, END_SEQ + 1):

    print(f"Processing sequence {seq_number}/{END_SEQ}.")

    # Load the first sequence.
    # seq_number = 1
    seq_filename = os.path.join(PATH_TO_DATASET, "data", f"sequence_{seq_number}", "scenes.json")
    sequence = Sequence.from_json(seq_filename)

    # Compute the frames for the sequence.
    timestamps = sequence.timestamps
    frames = group_timestamps_by_time(timestamps, 500)

    i = 0
    # Aggregate the point clouds.
    for frame in track(frames, description="Frame loop..."):
        
        i += 1
        # print(f"{'*' * 20} Frame {i} {'*' * 20}\n")

        apc = aggregated_point_cloud(frame, sequence)
        point_cloud = apc[0]
        # print(point_cloud)

        # Filter the point cloud.
        # Remove points that are out of the image range.
        point_cloud = point_cloud[(point_cloud["x_mod"] > -50) & (point_cloud["x_mod"] < 50)]
        point_cloud = point_cloud[(point_cloud["y_mod"] > -25) & (point_cloud["y_mod"] < 75)]
        # Remove the point whose label_id is 9 or 10.
        point_cloud = point_cloud[(point_cloud["label_id"] != 9) & (point_cloud["label_id"] != 10)]
        # print(point_cloud)

        # Feature scaling
        point_cloud["rcs_feat"] = point_cloud.apply(lambda x: rcs_scale(x['rcs']), axis=1)
        point_cloud["vr_feat"] = point_cloud.apply(lambda x: vr_scale(x['vr_compensated']), axis=1)

        # Extract the objects.
        track_ids = point_cloud['track_id'].unique().tolist()
        objects = [obj for obj in track_ids if len(obj) > 3]
        # print(objects, "\n")

        # # If the folder for the sequence does not exist, create it.
        # if not os.path.exists(os.path.join(PATH_TO_NEW_DATASET, f"sequence_{seq_number}")):
        #     os.makedirs(os.path.join(PATH_TO_NEW_DATASET, f"sequence_{seq_number}"))

        # --- CREATE A GRID MAP FROM AGGREGATED POINT
        N_CELLS = 640
        CELL_SIZE = 0.16

        # Create an empty grid
        grid = np.zeros((4, N_CELLS, N_CELLS))
        grid[0:3,:,:].fill(0.5)

        # Add every point to the grid
        for row in point_cloud.itertuples():
            if -50 <= row.x_mod <= 50 and -25 <= row.y_mod <= 75:
                x = ((row.x_mod / CELL_SIZE) + N_CELLS / 2)
                y = 640 - ((row.y_mod / CELL_SIZE) + N_CELLS / 4)
                if 0 <= x < N_CELLS and 0 <= y < N_CELLS:
                    grid[0, int(y), int(x)] = max(grid[0, int(y), int(x)], row.rcs_feat, 0.5)
                    grid[1, int(y), int(x)] = max(grid[1, int(y), int(x)], row.vr_feat, 0.5)
                    grid[2, int(y), int(x)] = min(grid[2, int(y), int(x)], row.vr_feat, 0.5)
                    grid[3, int(y), int(x)] += 1        

        # Change grid from N x W x H to W x H x N
        grid[grid == 0.5] = 0.0
        grid = np.transpose(grid, (1, 2, 0))

        KERNEL_SIZE = (7, 7)
        img_wo_blur = grid[:,:,0:3]
        img_blur = cv.GaussianBlur(img_wo_blur, KERNEL_SIZE, 0)

        # Multiply channels by 255 to get them in the range [0, 255].
        img_blur = img_blur * 255

        # Save the image
        if f"sequence_{seq_number}" in training_sequences:      # Training set
            data_filename = os.path.join(
                PATH_TO_NEW_DATASET,
                f"images",
                f"train",
                f"frame_{seq_number:03d}_{i:03d}.png"
            )
        else:                                                   # Validation set
            data_filename = os.path.join(
                PATH_TO_NEW_DATASET,
                f"images",
                f"val",
                f"frame_{seq_number:03d}_{i:03d}.png"
            )
        cv.imwrite(data_filename, img_blur)