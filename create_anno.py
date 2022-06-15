import os

from preprocessing import group_timestamps_by_time, aggregated_point_cloud, \
    category_map
from radar_scenes.sequence import Sequence

# Load RadarScenes dataset.
PATH_TO_DATASET = "/home/robesafe/Datasets/RadarScenes"
if not os.path.exists(PATH_TO_DATASET):
    raise FileNotFoundError("Dataset not found at {}".format(PATH_TO_DATASET))

# Set the path to the new dataset.
PATH_TO_NEW_DATASET = "/home/robesafe/Datasets/RadarScenes_BEV"
if not os.path.exists(PATH_TO_NEW_DATASET):
    os.makedirs(PATH_TO_NEW_DATASET)

# Load the first sequence.
seq_number = 1
seq_filename = os.path.join(PATH_TO_DATASET, "data", f"sequence_{seq_number}", "scenes.json")
sequence = Sequence.from_json(seq_filename)

# Compute the frames for the sequence.
timestamps = sequence.timestamps
frames = group_timestamps_by_time(timestamps, 500)

# Aggregate the point clouds.
for frame in frames:
    
    apc = aggregated_point_cloud(frame, sequence)
    point_cloud = apc[0]
    print(point_cloud)

    # Filter the point cloud.
    # Remove points that are out of the image range.
    point_cloud = point_cloud[(point_cloud["x_mod"] > -50) & (point_cloud["x_mod"] < 50)]
    point_cloud = point_cloud[(point_cloud["y_mod"] > -25) & (point_cloud["y_mod"] < 75)]
    # Remove the point whose label_id is 9 or 10.
    point_cloud = point_cloud[(point_cloud["label_id"] != 9) & (point_cloud["label_id"] != 10)]
    print(point_cloud)

    # Extract the objects.
    track_ids = point_cloud['track_id'].unique().tolist()
    objects = [obj for obj in track_ids if len(obj) > 3]
    print(objects, "\n")

    for obj in objects:
        print(f"------------------- {obj} -------------------")
        
        obj_points = point_cloud[point_cloud['track_id'] == obj]

        # Get the category of the object.
        obj_category = category_map.get(obj_points['label_id'].values[0])
        print(obj_category)

        # Get the bounding box.
        x0 = obj_points['x_mod'].min()
        y0 = obj_points['y_mod'].min()
        x1 = obj_points['x_mod'].max()
        y1 = obj_points['y_mod'].max()
        # print(x0, y0, x1, y1)
        print(f"Bounding box in map coords: {x0}, {y0}, {x1}, {y1}")

        # Map it to YOLO format.
        N_CELLS = 640
        CELL_SIZE = 0.16
        x0_cell = ((x0 / CELL_SIZE) + N_CELLS / 2) - 1
        y0_cell = 640 - ((y0 / CELL_SIZE) + N_CELLS / 4) - 1
        x1_cell = ((x1 / CELL_SIZE) + N_CELLS / 2) - 1
        y1_cell = 640 - ((y1 / CELL_SIZE) + N_CELLS / 4) - 1
        print(f"Bounding box in camera coords: {x0_cell}, {y0_cell}, {x1_cell}, {y1_cell}")

        xc_yolo = (x1_cell + x0_cell) / 2 / N_CELLS
        yc_yolo = (y1_cell + y0_cell) / 2 / N_CELLS
        w_yolo = (x1_cell - x0_cell) / N_CELLS
        h_yolo = - (y1_cell - y0_cell) / N_CELLS
        print(f"Bounding box in YOLO format: {xc_yolo}, {yc_yolo}, {w_yolo}, {h_yolo}")

    break