import numpy as np


def gen_hash_key(patch: np.ndarray, angle_base, coherence_base, strength_base, weight):
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

    # Compute the coherence
    sqrt_lambda1 = np.sqrt(w[0])
    sqrt_lambda2 = np.sqrt(w[1])
    if sqrt_lambda1 + sqrt_lambda2 == 0:
        u = 0
    else:
        u = (sqrt_lambda1 - sqrt_lambda2) / (sqrt_lambda1 + sqrt_lambda2)

    # Compute the theta of the eigenvector corresponding to the largest eigenvalue
    theta = np.arctan2(v[0, 1], v[0, 0])
    if theta < 0:
        theta += np.pi

    # Transform to integer via base
    angle = np.floor(theta / np.pi * angle_base) % angle_base
    coherence = np.floor(u * coherence_base) % coherence_base
    strength = np.floor(w[0] / np.sum(w) * strength_base) % strength_base

    if np.isnan(angle):
        angle = 0
    if np.isnan(coherence):
        coherence = 0
    if np.isnan(strength):
        strength = 0

    return int(angle), int(coherence), int(strength)
