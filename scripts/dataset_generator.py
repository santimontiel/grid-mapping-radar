""" RadarScenes-BEV dataset generation.
"""

# Python Standard library imports.
import argparse
import os
import sys # Add previous folder to the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Pip-requiring imports.
import numpy as np
import pandas as pd

# Module imports.
from utils.preprocessing import group_timestamps_by_time, aggregated_point_cloud, \
    category_num_map, category_sem_map

# RadarScenes imports.
from radar_scenes.sequence import Sequence, get_training_sequences, \
    get_validation_sequences

class Logger:
    def __init__(self) -> None:
        pass

class DatasetGenerator:
    def __init__(self, old_path, new_path) -> None:
        self._OLD_PATH = old_path
        self._NEW_PATH = new_path
        self._logger = Logger()

    def run():
        pass

    def loop():
        pass

    def point_cloud_preprocessing(point_cloud: pd.DataFrame) -> pd.DataFrame:
        """
        """
        pass

    def image_preprocessing(point_cloud: pd.DataFrame) -> np.array:
        pass

    def generate_annotations_for_image():
        pass

    def generate_image():
        pass

def argparser():
    pass

def main(args=None):
    """ Main function for RadarScenes-BEV generator.
    """

    # TODO: Set parameters with argparser and/or YAML config file.

    # Set the path of RadarScenes in your machine.
    OLD_PATH = "/home/robesafe/Datasets/RadarScenes"
    if not os.path.exists(OLD_PATH):
        raise FileNotFoundError("RadarScenes not found at {}".format(OLD_PATH))

    # Set the path for the new dataset and create folder structure.
    NEW_PATH = "/home/robesafe/Santi/test_radarscenes_bev"
    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH)
    os.makedirs(os.path.join(NEW_PATH, "images", "train"))
    os.makedirs(os.path.join(NEW_PATH, "images", "val"))
    os.makedirs(os.path.join(NEW_PATH, "labels", "train"))
    os.makedirs(os.path.join(NEW_PATH, "labels", "val"))

    # Get the training and validation sequences.
    training_sequences = get_training_sequences(os.path.join(OLD_PATH, "data", "sequences.json"))
    validation_sequences = get_validation_sequences(os.path.join(OLD_PATH, "data", "sequences.json"))

if __name__ == "__main__":

    main()