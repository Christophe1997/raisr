import numpy as np
from typing import Tuple


class HashKeyGen:

    def __init__(self, angle_base=24, strength_base=3, coherence_base=3):
        self.angle_base = angle_base
        self.strength_base = strength_base
        self.coherence_base = coherence_base

    def __repr__(self):
        return f"<HashKeyGen ({self.angle_base}, {self.strength_base}, {self.coherence_base})>"

    def gen_hash_key(self, image_block: np.ndarray, weight: np.ndarray) -> Tuple[int, int, int]:
        g_y, g_x = np.gradient(image_block)
        g_x: np.ndarray = g_x.ravel()
        g_y: np.ndarray = g_y.ravel()

        G: np.ndarray = np.vstack((g_x, g_y)).T
        GT_W_G = np.dot(G.T, weight).dot(G)
        w, v = np.linalg.eig(GT_W_G)
        w = np.real(w)
        v = np.real(v)

        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:, idx]

        sqrt_lambda1 = np.sqrt(w[0])
        sqrt_lambda2 = np.sqrt(w[1])
        if sqrt_lambda1 + sqrt_lambda2 == 0:
            u = 0
        else:
            # compute the coherence
            u = (sqrt_lambda1 - sqrt_lambda2) / (sqrt_lambda1 + sqrt_lambda2)

        # compute the theta of the eigenvector corresponding to the largest eigenvalue
        theta = np.arctan2(v[0, 1], v[0, 0])
        if theta < 0:
            theta += np.pi

        angle = np.floor(theta / np.pi * self.angle_base)
        coherence = np.floor(u * self.coherence_base)
        strength = np.floor((w[0] / np.sum(w)) * self.strength_base)

        return int(angle), int(coherence), int(strength)
