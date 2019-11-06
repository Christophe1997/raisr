import cv2 as cv
import numpy as np


def gaussian_kernel_2d(shape, sigma=0.5) -> np.ndarray:
    if len(shape) != 2:
        raise Exception("Wrong shape")
    x: np.ndarray = cv.getGaussianKernel(shape[0], sigma)
    y: np.ndarray = cv.getGaussianKernel(shape[1], sigma)

    return x.dot(y.T)


UNSHARP_MASKING_5x5_KERNEL = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, -476, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]) / -256
