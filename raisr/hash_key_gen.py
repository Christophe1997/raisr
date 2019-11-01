import numpy as np
from typing import Tuple


class HashKeyGen:
    """Hash key generator
    It compute the angle, strength and coherence of local gradient as the hash key
    """

    def __init__(self, weight: np.ndarray,
                 angle_base: int = 24, strength_base: int = 3, coherence_base: int = 3):
        self.weight = weight
        self.angle_base = angle_base
        self.strength_base = strength_base
        self.coherence_base = coherence_base

    def __repr__(self):
        return f"<HashKeyGen ({self.angle_base}, {self.strength_base}, {self.coherence_base})>"

    def gen_hash_key(self, patch: np.ndarray, ) -> Tuple[int, int, int]:
        # Compute the gradient
        g_y, g_x = np.gradient(patch)
        g_x: np.ndarray = g_x.ravel()
        g_y: np.ndarray = g_y.ravel()

        G: np.ndarray = np.vstack((g_x, g_y)).T
        GT_W_G = np.dot(G.T, self.weight).dot(G)
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
        angle = np.floor(theta / np.pi * self.angle_base)
        coherence = np.floor(u * self.coherence_base)
        strength = np.floor((w[0] / np.sum(w)) * self.strength_base)

        return int(angle), int(coherence), int(strength)
