# --- Imports
import os
from radar_scenes.sequence import Sequence, get_training_sequences
from preprocessing import group_timestamps_by_time, aggregated_point_cloud, \
    plot_aggregated_point_cloud

# 1. Load RadarScenes dataset.
PATH_TO_DATASET = "/home/robesafe/Datasets/RadarScenes"
if not os.path.exists(PATH_TO_DATASET):
    raise FileNotFoundError("Dataset not found at {}".format(PATH_TO_DATASET))

# 2. Load the first sequence.
seq_number = 1
seq_filename = os.path.join(PATH_TO_DATASET, "data", f"sequence_{seq_number}", "scenes.json")
sequence = Sequence.from_json(seq_filename)

# 3. Generate the frames for the sequence.
frames = group_timestamps_by_time(sequence.timestamps, 500)
print(frames)

# 4. Iterate over the frames and aggregate the point clouds.
for frame in frames:
    apc = aggregated_point_cloud(frame, sequence)
    point_cloud = apc[0]
    last_image = apc[1]
    x0 = float(apc[2])
    y0 = float(apc[3])
    x1 = float(apc[4])
    y1 = float(apc[5])
    plot_aggregated_point_cloud(point_cloud, last_image, x0, y0, x1, y1)