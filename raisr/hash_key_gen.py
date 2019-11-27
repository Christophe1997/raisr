import numpy as np


def gen_hash_key(patch: np.ndarray, angle_base, strength_base):
    alpha = 0.95
    # Compute the gradient
    g_y, g_x = np.gradient(patch)
    g_x: np.ndarray = g_x.ravel()
    g_y: np.ndarray = g_y.ravel()

    G: np.ndarray = np.vstack((g_x, g_y)).T
    u, s, _ = np.linalg.svd(G.T)

    theta = np.arctan2(u[0, 0], u[1, 0])
    if theta < 0:
        theta += np.pi

    # Transform to integer via base
    angle = np.floor(theta / np.pi * angle_base) % angle_base
    if np.sum(s) == 0:
        strength = 0
    else:
        strength = np.floor(alpha * s[0] / np.sum(s) * strength_base) % strength_base

    return int(angle), int(strength)
