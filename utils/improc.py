import numpy as np
import cv2 as cv

def apply_nxn_blur(grid: np.array, n: int):
    """
    """
    KERNEL_SIZE = (n, n)
    grid = cv.GaussianBlur(grid, KERNEL_SIZE, 0)
    return grid

def apply_opening(grid: np.array) -> np.array:
    """
    """
    (b, g, r) = cv.split(grid)

    opened_chs = []
    for i in (b, g, r):
        opened_chs.append(cv.morphologyEx(i, cv.MORPH_OPEN, (3,3)))
    merged = cv.merge([opened_chs[0], opened_chs[1], opened_chs[2]])
    return merged
