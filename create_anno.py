import os
from rich.progress import track

from preprocessing import group_timestamps_by_time, aggregated_point_cloud, \
    category_num_map, category_sem_map
from radar_scenes.sequence import Sequence, get_training_sequences, \
    get_validation_sequences

# Load RadarScenes dataset.
PATH_TO_DATASET = "/home/robesafe/Datasets/RadarScenes"
if not os.path.exists(PATH_TO_DATASET):
    raise FileNotFoundError("Dataset not found at {}".format(PATH_TO_DATASET))

# Set the path to the new dataset.
PATH_TO_NEW_DATASET = "/home/robesafe/Santi/radarscenes_bev_yolo_v4"
if not os.path.exists(PATH_TO_NEW_DATASET):
    os.makedirs(PATH_TO_NEW_DATASET)
os.makedirs(os.path.join(PATH_TO_NEW_DATASET, "labels", "train"))
os.makedirs(os.path.join(PATH_TO_NEW_DATASET, "labels", "val"))

# Get the training sequences.
training_sequences = get_training_sequences(os.path.join(PATH_TO_DATASET, "data", "sequences.json"))
validation_sequences = get_validation_sequences(os.path.join(PATH_TO_DATASET, "data", "sequences.json"))

# Number of sequences
START_SEQ, END_SEQ = 1, 158

for seq_number in range(START_SEQ, END_SEQ + 1):

    print(f"Processing sequence {seq_number}/{END_SEQ}.")

    # Load the sequence.
    seq_filename = os.path.join(PATH_TO_DATASET, "data", f"sequence_{seq_number}", "scenes.json")
    sequence = Sequence.from_json(seq_filename)

    # Compute the frames for the sequence.
    timestamps = sequence.timestamps
    frames = group_timestamps_by_time(timestamps, 500)

    # Aggregate the point clouds.
    i = 0
    for frame in track(frames, description="Frame loop..."):
        
        i += 1

        apc = aggregated_point_cloud(frame, sequence)
        point_cloud = apc[0]


        # Filter the point cloud.
        # Remove points that are out of the image range.
        point_cloud = point_cloud[(point_cloud["x_mod"] > -50) & (point_cloud["x_mod"] < 50)]
        point_cloud = point_cloud[(point_cloud["y_mod"] > -25) & (point_cloud["y_mod"] < 75)]
        # Remove the point whose label_id is 9 or 10.
        point_cloud = point_cloud[(point_cloud["label_id"] != 9) & (point_cloud["label_id"] != 10)]

        # Extract the objects.
        track_ids = point_cloud['track_id'].unique().tolist()
        objects = [obj for obj in track_ids if len(obj) > 3]
        # print(objects, "\n")

        # # If the folder for the sequence does not exist, create it.
        # if not os.path.exists(os.path.join(PATH_TO_NEW_DATASET, f"sequence_{seq_number}")):
        #     os.makedirs(os.path.join(PATH_TO_NEW_DATASET, f"sequence_{seq_number}"))

        # Open the annotation file and write the objects.
        if f"sequence_{seq_number}" in training_sequences:      # Training set
            anno_filename = os.path.join(
                PATH_TO_NEW_DATASET,
                f"labels",
                f"train",
                f"frame_{seq_number:03d}_{i:03d}.txt"
            )
        else:                                                   # Validation set
            anno_filename = os.path.join(
                PATH_TO_NEW_DATASET,
                f"labels",
                f"val",
                f"frame_{seq_number:03d}_{i:03d}.txt"
            )


        with open(anno_filename, "w") as f:

            for obj in objects:
                # print(f"------------------- {obj} -------------------")
                
                obj_points = point_cloud[point_cloud['track_id'] == obj]

                # Get the category of the object.
                obj_num_cat = category_num_map.get(obj_points['label_id'].values[0])
                obj_sem_cat = category_sem_map.get(obj_points['label_id'].values[0])
                # print(obj_num_cat, obj_sem_cat)

                # Get the bounding box.
                x0 = obj_points['x_mod'].min()
                y0 = obj_points['y_mod'].min()
                x1 = obj_points['x_mod'].max()
                y1 = obj_points['y_mod'].max()
                # print(x0, y0, x1, y1)
                # print(f"Bounding box in map coords: {x0}, {y0}, {x1}, {y1}")

                # Map it to YOLO format.
                N_CELLS = 640
                CELL_SIZE = 0.16
                x0_cell = ((x0 / CELL_SIZE) + N_CELLS / 2) - 1
                y0_cell = 640 - ((y0 / CELL_SIZE) + N_CELLS / 4) - 1
                x1_cell = ((x1 / CELL_SIZE) + N_CELLS / 2) - 1
                y1_cell = 640 - ((y1 / CELL_SIZE) + N_CELLS / 4) - 1
                # print(f"Bounding box in camera coords: {x0_cell}, {y0_cell}, {x1_cell}, {y1_cell}")

                xc_yolo = (x1_cell + x0_cell) / 2 / N_CELLS
                yc_yolo = (y1_cell + y0_cell) / 2 / N_CELLS
                w_yolo = (x1_cell - x0_cell) / N_CELLS
                if w_yolo < 3/640:
                    w_yolo = 0.01
                h_yolo = - (y1_cell - y0_cell) / N_CELLS
                if h_yolo < 3/640:
                    h_yolo = 0.01

                # print(f"Bounding box in YOLO format: {xc_yolo}, {yc_yolo}, {w_yolo}, {h_yolo}")

                # Write the annotation.
                f.write(f"{obj_num_cat} {xc_yolo:.5f} {yc_yolo:.5f} {w_yolo:.5f} {h_yolo:.5f}\n")

        # Close and save the file
        f.close()
        # print(f"Saved annotation file for frame {i} to {anno_filename}")