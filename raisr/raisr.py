import cv2 as cv
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.image import extract_patches_2d
from raisr.utils import is_image
import os
import pickle
from raisr.hash_key_gen import HashKeyGen
from raisr.processor import UNSHARP_MASKING_5x5_KERNEL
from datetime import datetime
import time


class RAISR:

    def __init__(self, upscale_factor=2,
                 angle_base=24, strength_base=3, coherence_base=3):
        """RAISR model

        :param upscale_factor: upsampling factor;
        :param angle_base: factor for angle
        :param strength_base: factor for strength
        :param coherence_base: factor for coherence
        """
        self.up_factor = upscale_factor
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

    def preprocess(self, lr_path: str, hr_path: str,
                   patch_size: int = 7,
                   dst: str = "./train",
                   sharpen: bool = True,
                   ):
        """Process origin images, both of LR images and HR images
        It only extrat the luminance in YCrCb mode of a image, it also cheaply upscale and pad the LR images
        for trainning requirements. And write the result to dst.
        :param sharpen: If true, use unsharp masking kernel to enhance HR images
        :param lr_path: LR images' dir path,
        :param hr_path: HR images' dir path,
        :param patch_size: filter size, always it should same as which in train step.
        :param dst: Where to save the result file,
        :return: void, use pickle to write [(LR, HR)...] to dst.
        """

        # It depends on that LR image with the associated HR image have the same prefix
        # otherwise, it may cause wrong result
        lr_files = sorted(filter(is_image, os.listdir(lr_path)))
        hr_filrs = sorted(filter(is_image, os.listdir(hr_path)))

        # compute pad pixel
        # left is same as top, and right is same as bottom
        top_pad = (patch_size - 1) // 2
        if patch_size % 2 == 0:
            bottom_pad = top_pad + 1
        else:
            bottom_pad = top_pad

        ret = []
        total = len(lr_files)
        print(f"*****START TO PROCESS {total} images*****\n")
        for idx, lr_fname in enumerate(lr_files):
            print("*****START TO PROCESS {}/{} image at {}*****".format(
                idx, total, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            hr_fname = hr_filrs[idx]
            lr = cv.imread(os.path.join(lr_path, lr_fname))
            hr = cv.imread(os.path.join(hr_path, hr_fname))
            # Extrat the luminance in YCrCb mode of a image
            lr_y = cv.cvtColor(lr, cv.COLOR_BGR2YCrCb)[:, :, 0]
            hr_y = cv.cvtColor(hr, cv.COLOR_BGR2YCrCb)[:, :, 0]
            # Upscale and pad the image
            h, w = lr_y.shape
            lr_y_upscaled_padded = cv.copyMakeBorder(cv.resize(lr_y, (w * self.up_factor, h * self.up_factor)),
                                                     top_pad, bottom_pad, top_pad, bottom_pad, cv.BORDER_REPLICATE)
            # optionally sharpen
            if sharpen:
                hr_y = cv.filter2D(hr_y, -1, UNSHARP_MASKING_5x5_KERNEL, borderType=cv.BORDER_REPLICATE)
            ret.append((lr_y_upscaled_padded, hr_y))

            print("*****END   TO PROCESS {}/{} image at {}*****\n".format(
                idx, total, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        timestamp = time.mktime(datetime.now().timetuple())
        if not os.path.exists(dst):
            os.mkdir(dst)
        dump_path = os.path.join(dst, "raisr_train_data_{:.0f}.pkl".format(timestamp))
        with open(dump_path, "wb") as f:
            pickle.dump(ret, f)

        print("*****FINISH PROCESS, RESULT DUMP TO {}*****\n".format(dump_path))

    def cheap_upscale(self, image: np.ndarray, interpolation=cv.INTER_LINEAR) -> np.ndarray:
        hight, width = image.shape[:2]
        return cv.resize(image, (width * self.up_factor, hight * self.up_factor), interpolation=interpolation)

    def __compute_j(self, j: Tuple[int, int, int]):
        a, c, s = j  # angle, coherence, strength
        y = c * self.coherence_base + s
        return a, y
