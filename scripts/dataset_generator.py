""" RadarScenes-BEV dataset generation.

WIP... @TODO LIST.
1. Multiprocessing for faster processing.
2. Explore new feature scaling and image processing.
3. Logger.
"""

# Python Standard library imports.
import argparse
import os
import sys # Add previous folder to the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List

# Pip-requiring imports.
import numpy as np
import pandas as pd
import cv2 as cv

# Module imports.
from utils.preprocessing import group_timestamps_by_time, aggregated_point_cloud, \
    category_num_map, category_sem_map
from utils.filters import rcs_scale, vr_scale
from utils.improc import apply_opening, apply_nxn_blur

# RadarScenes imports.
from radar_scenes.sequence import Sequence, get_training_sequences, \
    get_validation_sequences

class Logger:
    """ @TODO: Nice and fancy logger. """
    def __init__(self) -> None:
        pass

class DatasetGenerator:
    """
    """

    def __init__(self, old_path: str, new_path: str) -> None:

        self._OLD_PATH = old_path
        self._NEW_PATH = new_path
        self._seq_cnt, self._sce_cnt, self._fra_cnt = 0, 0, 0 # Sequence, scene, frame
        self._SEQ_START, self._SEQ_END = 1, 158 # Dataset has 158 sequences, numbered from 1 to 158.
        self._N_CELLS = 640
        self._CELL_SIZE = 0.16
        # @TODO: self._logger = Logger()


        # -- LOAD DATASET AND PREPARE FOLDER STRUCTURE
        # Check if the dataset path exists.
        if not os.path.exists(self._OLD_PATH):
            raise FileNotFoundError("RadarScenes not found at {}".format(self._OLD_PATH))

        # Prepare the folder structure for the new path.
        if not os.path.exists(self._NEW_PATH):
            os.makedirs(self._NEW_PATH)
        os.makedirs(os.path.join(self._NEW_PATH, "images", "train"))
        os.makedirs(os.path.join(self._NEW_PATH, "images", "val"))
        os.makedirs(os.path.join(self._NEW_PATH, "labels", "train"))
        os.makedirs(os.path.join(self._NEW_PATH, "labels", "val"))

        # Get the training and validation sequences.
        self.train_seqs = get_training_sequences(os.path.join(self._OLD_PATH, "data", "sequences.json"))
        self.val_seqs = get_validation_sequences(os.path.join(self._OLD_PATH, "data", "sequences.json"))

    def run(self):
        """
        """

        # Iterate through each sequence of the dataset.
        for n_seq in range(self._SEQ_START, self._SEQ_END + 1):

            print(f"Sequence {n_seq} of {self._SEQ_END}")

            # Load the n-th sequence.
            seq_filename = os.path.join(self._OLD_PATH, "data", f"sequence_{n_seq}", "scenes.json")
            sequence = Sequence.from_json(seq_filename)

            # Compute the frame for this sequence.
            timestamps = sequence.timestamps
            frames = group_timestamps_by_time(timestamps, 500)

            # Iterate through each frame of the sequence.
            for n_frame, frame in enumerate(frames):

                # Generate the aggregated point cloud.
                apc = aggregated_point_cloud(frame, sequence)
                point_cloud = apc[0]

                # Extract the objects in the point cloud.
                track_ids = list(point_cloud['track_id'].unique())
                objects = [obj for obj in track_ids if len(obj) > 3]

                # Feature scaling. -> USER CUSTOMIZABLE
                try:
                    point_cloud["rcs_feat"] = point_cloud.apply(lambda x: rcs_scale(x['rcs']), axis=1)
                    point_cloud["vr_feat"] = point_cloud.apply(lambda x: vr_scale(x['vr_compensated']), axis=1)
                except ValueError:  # This happens when DataFrame is empty (1 case in the dataset -> sequence 154)
                    continue        # just skip that frame

                # Grid map generation and image preprocessing. -> USER CUSTOMIZABLE
                grid = self.image_preprocessing(point_cloud)

                # Grid custom processing. -> USER CUSTOMIZABLE
                grid = apply_nxn_blur(grid, 3)

                # Filenames for annotations and images.
                train_or_val = "train" if f"sequence_{n_seq}" in self.train_seqs else "val"
                image_filename = os.path.join(self._NEW_PATH, f"images", train_or_val, f"frame_{n_seq:03d}_{n_frame:03d}.png")
                anno_filename = os.path.join(self._NEW_PATH, f"labels", train_or_val, f"frame_{n_seq:03d}_{n_frame:03d}.txt")

                # Generate annotations for the image.
                self.generate_annotations_for_image(anno_filename, point_cloud, objects)

                # Generate the final image.
                self.generate_image(image_filename, grid)

    def image_preprocessing(self, point_cloud: pd.DataFrame) -> np.array:
        """
        """

        # Create an empty grid
        grid = np.zeros((4, self._N_CELLS, self._N_CELLS))
        grid[0:3,:,:].fill(0.5)

        # Add every point to the grid
        for row in point_cloud.itertuples():
            if -50 <= row.x_mod <= 50 and -25 <= row.y_mod <= 75:
                x = ((row.x_mod / self._CELL_SIZE) + self._N_CELLS / 2)
                y = 640 - ((row.y_mod / self._CELL_SIZE) + self._N_CELLS / 4)
                if 0 <= x < self._N_CELLS and 0 <= y < self._N_CELLS:
                    grid[0, int(y), int(x)] = max(grid[0, int(y), int(x)], row.rcs_feat, 0.5)
                    grid[1, int(y), int(x)] = max(grid[1, int(y), int(x)], row.vr_feat, 0.5)
                    grid[2, int(y), int(x)] = min(grid[2, int(y), int(x)], row.vr_feat, 0.5)
                    grid[3, int(y), int(x)] += 1 # @TODO: Cell propagation scheme.       

        # Change grid from N x W x H to W x H x N
        grid = np.transpose(grid, (1, 2, 0))
        grid = grid[:, :, :3]
        grid[grid == 0.5] = 0.0
        grid *= 255.0

        return grid

    def generate_annotations_for_image(self, anno_filename: str, point_cloud: pd.DataFrame, objects: List):
        """
        """
        
        with open(anno_filename, "w") as f:

            for obj in objects:
                
                obj_points = point_cloud[point_cloud['track_id'] == obj]

                # Get the category of the object.
                obj_num_cat = category_num_map.get(obj_points['label_id'].values[0])
                obj_sem_cat = category_sem_map.get(obj_points['label_id'].values[0])

                # Get the bounding box.
                x0 = obj_points['x_mod'].min()
                y0 = obj_points['y_mod'].min()
                x1 = obj_points['x_mod'].max()
                y1 = obj_points['y_mod'].max()

                # Map it to YOLO format.
                N_CELLS = 640
                CELL_SIZE = 0.16
                x0_cell = ((x0 / CELL_SIZE) + N_CELLS / 2) - 1
                y0_cell = 640 - ((y0 / CELL_SIZE) + N_CELLS / 4) - 1
                x1_cell = ((x1 / CELL_SIZE) + N_CELLS / 2) - 1
                y1_cell = 640 - ((y1 / CELL_SIZE) + N_CELLS / 4) - 1

                xc_yolo = (x1_cell + x0_cell) / 2 / N_CELLS
                yc_yolo = (y1_cell + y0_cell) / 2 / N_CELLS
                w_yolo = (x1_cell - x0_cell) / N_CELLS
                if w_yolo < 3/640:
                    w_yolo = 0.01
                h_yolo = - (y1_cell - y0_cell) / N_CELLS
                if h_yolo < 3/640:
                    h_yolo = 0.01

                # Write the annotation.
                f.write(f"{obj_num_cat} {xc_yolo:.5f} {yc_yolo:.5f} {w_yolo:.5f} {h_yolo:.5f}\n")

        # Close and save the file
        f.close()

    def generate_image(self, image_filename: str, grid: np.array):
        """
        """
        cv.imwrite(image_filename, grid)


def argparser():
    pass

def main(args=None):
    """ Main function for RadarScenes-BEV generator.
    """

    # @TODO: Set parameters with argparser and/or YAML config file.

    # Set the paths: RadarScenes in your machine and destination folder.
    # ... Lab computer
    # OLD_PATH = "/home/robesafe/Datasets/RadarScenes"
    # NEW_PATH = "/home/robesafe/Santi/test_radarscenes_bev"
    # ... Home computer
    OLD_PATH = "C:\\Datasets\\RadarScenes"
    NEW_PATH = "C:\\Datasets\\RadarScenes_BEV_GaussianBlur3x3"
    
    # Initialize the Dataset Generator object.
    dataset_generator = DatasetGenerator(
        old_path=OLD_PATH,
        new_path=NEW_PATH,
    )

    # Run the generator.
    dataset_generator.run()

    

if __name__ == "__main__":

    main()