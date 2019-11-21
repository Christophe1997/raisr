import numpy as np


def gen_hash_key(patch: np.ndarray, angle_base, weight) -> int:
    # Compute the gradient
    g_y, g_x = np.gradient(patch)
    g_x: np.ndarray = g_x.ravel()
    g_y: np.ndarray = g_y.ravel()

    G: np.ndarray = np.vstack((g_x, g_y)).T
    GT_W_G = np.dot(G.T, weight).dot(G)
    w, v = np.linalg.eig(GT_W_G)
    w = np.real(w)
    v = np.real(v)

    # Sort the eigenvalue and eigenvector
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]

    # Compute the theta of the eigenvector corresponding to the largest eigenvalue
    theta = np.arctan2(v[0, 1], v[0, 0])
    if theta < 0:
        theta += np.pi

    # Transform to integer via base
    angle = np.floor(theta / np.pi * angle_base) % angle_base

    if np.isnan(angle):
        angle = 0

    return int(angle)
