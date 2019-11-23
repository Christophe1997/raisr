import numpy as np


def is_image(fp_path: str):
    return fp_path.lower().endswith(('.png', '.jpg', '.tiff', '.bmp', '.gif'))


def build_goal_func(Q, V):
    def _(h):
        return np.linalg.norm(np.dot(Q, h) - V) ** 2

    return _
