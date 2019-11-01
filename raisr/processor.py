import cv2 as cv
import numpy as np


def gaussian_kernel(shape, sigma=0.5) -> np.ndarray:
    if len(shape) != 2:
        raise Exception("Wrong shape")
    x: np.ndarray = cv.getGaussianKernel(shape[0], sigma)
    y: np.ndarray = cv.getGaussianKernel(shape[1], sigma)

    return x.dot(y.T)
