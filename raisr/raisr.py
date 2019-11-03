import cv2 as cv
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.image import extract_patches_2d
import pickle
from raisr.hash_key_gen import HashKeyGen


class RAISR:

    def __init__(self, upscale_factor=2,
                 enhance_hr=None, enhance_lr=None,
                 angle_base=24, strength_base=3, coherence_base=3):
        """RAISR model

        :param upscale_factor: upsampling factor;
        :param enhance_hr: method that used to enhance HR image
        :param enhance_lr: method that used to enhance LR image
        :param angle_base: factor for angle
        :param strength_base: factor for strength
        :param coherence_base: factor for coherence
        """
        self.up_factor = upscale_factor
        self.enhance_hr = enhance_hr
        self.enhance_lr = enhance_lr
        self.angle_base = angle_base
        self.strength_base = strength_base
        self.coherence_base = coherence_base

    def __repr__(self):
        return f"<RAISR up_factor={self.up_factor}>"

    def train(self, img_pairs, patch_size=7):
        # initialization
        d2 = patch_size ** 2
        t2 = self.up_factor ** 2
        axis0 = self.angle_base
        axis1 = self.coherence_base * self.strength_base

        Q = np.zeros((axis0, axis1, t2, d2, d2))
        V = np.zeros((axis0, axis1, t2, d2, 1))
        H = np.zeros((axis0, axis1, t2, patch_size, patch_size))

        # TODO: can be paralleled
        for cheap_hr_padded, hr in img_pairs:
            h, w = hr.sahpe
            patchs = extract_patches_2d(cheap_hr_padded, (patch_size, patch_size))
            A = None
            b = None
            # TODO: can be paralleled
            for idx, patch in enumerate(patchs):
                # origin coordinate
                x = idx // w
                y = idx % w

                # compute pixel type
                pixel_type = x % self.up_factor * self.up_factor + y % self.up_factor

                patch: np.ndarray

    def preprocess(self, lr_path, hr_path, patch_size=7, dst="./train"):
        """Process origin images, both of LR images and HR images
        It only extrat the luminance in YCbCr mode of a image, it also cheaply upscale and pad the LR images
        for trainning requirements. And write the result to dst.
        :param lr_path: LR images path,
        :param hr_path: HR images path,
        :param patch_size: filter size, always it should same as which in train step.
        :param dst: Where to save the result file,
        :return: void, use pickle to write [(LR, HR)...] to dst.
        """

        # compute pad pixel
        # left is same as top, and right is same as bottom
        top_pad = (patch_size - 1) // 2
        if patch_size % 2 == 0:
            bottom_pad = top_pad + 1
        else:
            bottom_pad = top_pad

    def cheap_upscale(self, image: np.ndarray, interpolation=cv.INTER_LINEAR) -> np.ndarray:
        hight, width = image.shape[:2]
        return cv.resize(image, (width * self.up_factor, hight * self.up_factor), interpolation=interpolation)

    def __compute_j(self, j: Tuple[int, int, int]):
        a, c, s = j  # angle, coherence, strength
        y = c * self.coherence_base + s
        return a, y
