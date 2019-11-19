import cv2 as cv
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from raisr.utils import is_image
import os
import pickle
from raisr.hash_key_gen import HashKeyGen
from raisr.processor import UNSHARP_MASKING_5x5_KERNEL, gaussian_kernel_2d
from datetime import datetime
import time
from collections import defaultdict
from scipy.sparse.linalg import cg
from pathos import multiprocessing as mp


class RAISR:

    def __init__(self, upscale_factor=2,
                 angle_base=24, strength_base=3, coherence_base=3, patch_size=7):
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
        self.patch_size = patch_size
        weight = gaussian_kernel_2d((patch_size ** 2, patch_size ** 2))
        self.hash_key_gen = HashKeyGen(weight, angle_base, strength_base, coherence_base)
        self.H = None

    def __repr__(self):
        return f"<RAISR up_factor={self.up_factor}>"

    def train(self, lr_path: str, hr_path: str, sharpen: bool = True, dst="./model"):
        """The train step of RAISR model, where RAISR learn a set of filters from data
        :param sharpen: If true, use unsharp masking kernel to enhance HR images,
        :param lr_path: LR images' dir path,
        :param hr_path: HR images' dir path,
        :param dst: Where to save the model,
        :return: void, use pickle to write H to `dst`.
        """
        # initialization
        d2 = self.patch_size ** 2
        t2 = self.up_factor ** 2
        axis0 = self.angle_base
        axis1 = self.coherence_base
        axis2 = self.strength_base

        Q = np.zeros((axis0, axis1, axis2, t2, d2, d2))
        V = np.zeros((axis0, axis1, axis2, t2, d2, 1))
        H = np.zeros((axis0, axis1, axis2, t2, d2, 1))

        # compute pad pixel
        # left is same as top, and right is same as bottom
        top_pad = (self.patch_size - 1) // 2
        if self.patch_size % 2 == 0:
            bottom_pad = top_pad + 1
        else:
            bottom_pad = top_pad

        lr_files = sorted(filter(is_image, os.listdir(lr_path)))
        hr_filrs = sorted(filter(is_image, os.listdir(hr_path)))

        total = len(lr_files)
        start = datetime.now()
        print(f"*****START TO TRAIN RAISR FOR {total} IMAGE PAIRS*****\n")

        for idx, lr_fname in enumerate(lr_files):
            print("*****START TO PROCESS {}/{} IMAGE PAIR AT {}*****".format(
                idx + 1, total, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            hr_fname = hr_filrs[idx]
            lr = cv.imread(os.path.join(lr_path, lr_fname))
            hr = cv.imread(os.path.join(hr_path, hr_fname))
            # Extrat the luminance in YCrCb mode of a image
            lr_y = cv.cvtColor(lr, cv.COLOR_BGR2YCrCb)[:, :, 0]
            hr_y = cv.cvtColor(hr, cv.COLOR_BGR2YCrCb)[:, :, 0]

            # standardlize
            lr_y = (lr_y - lr_y.mean()) / lr_y.std()
            hr_y = (hr_y - hr_y.mean()) / hr_y.std()
            # Upscale and pad the image
            h, w = lr_y.shape
            lr_y = cv.copyMakeBorder(
                cv.resize(lr_y, (w * self.up_factor, h * self.up_factor)),
                top_pad, bottom_pad, top_pad, bottom_pad, borderType=cv.BORDER_REPLICATE)
            # optionally sharpen
            if sharpen:
                hr_y = cv.filter2D(hr_y, -1, UNSHARP_MASKING_5x5_KERNEL, borderType=cv.BORDER_REPLICATE)

            h, w = hr_y.shape
            patches = extract_patches_2d(lr_y, (self.patch_size, self.patch_size))
            A = defaultdict(lambda: None)
            b = defaultdict(lambda: None)

            def f(item):
                idx1, patch = item
                # origin coordinate
                x = idx1 // w
                y = idx1 % w

                # compute pixel type
                t = x % self.up_factor * self.up_factor + y % self.up_factor
                # conpute hash key j
                j = self.hash_key_gen.gen_hash_key(patch)
                patch: np.ndarray
                # compute p_k
                p = patch.ravel().reshape(1, -1)
                # compute x_k, the true HR pixel
                v = hr[x, y]
                if A[j, t] is None:
                    A[j, t] = p
                else:
                    A[j, t] = np.vstack((A[j, t], p))
                if b[j, t] is None:
                    b[j, t] = v
                else:
                    b[j, t] = np.hstack((b[j, t], v))

            with mp.Pool(processes=mp.cpu_count()) as ps:
                ps.map(f, enumerate(patches))

            for j, t in A.keys():
                a_j_t = A[j, t]
                b_j_t = b[j, t]
                a_T_a = a_j_t.T.dot(a_j_t)
                a_T_b = a_j_t.T.dot(b_j_t)

                # increase date by flip and rotate
                rot90 = self.angle_base // 4
                for i in range(4):
                    angle = (j[0] + i * rot90) % self.angle_base
                    coherence, strength = j[1:]
                    Q[angle, coherence, strength, t] += a_T_a
                    V[angle, coherence, strength, t] += a_T_b
                    a_T_a = np.rot90(a_T_a)
                a_T_a = np.flipud(a_T_a)
                flipud = self.angle_base - j[0]
                for i in range(4):
                    angle = (flipud + i * rot90) % self.angle_base
                    coherence, strength = j[1:]
                    Q[angle, coherence, strength, t] += a_T_a
                    V[angle, coherence, strength, t] += a_T_b
                    a_T_a = np.rot90(a_T_a)

            print("*****END   TO PROCESS {}/{} IMAGE PAIR AT {}*****\n".format(
                idx + 1, total, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        print(f"*****START TO SOLVE THE OPTIMIZATION PROBLEM AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*****")
        # compute H
        for angle in range(axis0):
            for coherence in range(axis1):
                for strength in range(axis2):
                    for t in range(t2):
                        # solve the optimization problem by a conjugate gradient solver
                        h_vec, _ = cg(Q[angle, coherence, strength, t], V[angle, coherence, strength, t])
                        H[angle, coherence, strength, t] = h_vec.reshape((-1, 1))
        print(f"*****END   TO SOLVE THE OPTIMIZATION PROBLEM AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*****\n")

        # write the filter
        end = datetime.now()
        timestamp = time.mktime(end.timetuple())
        if not os.path.exists(dst):
            os.mkdir(dst)
        dump_path = os.path.join(dst, "raisr_filter_{}x{}_{}x{}x{}x{}_{:.0f}.pkl".format(
            self.patch_size, self.patch_size,
            axis0, axis1, axis2, t2,
            timestamp))
        with open(dump_path, "wb") as f:
            pickle.dump(H, f)

        self.H = H

        print("*****FINISH TRAIN, COSTS {}s, RESULT DUMP TO {}*****\n".format((end - start).total_seconds(), dump_path))

    def cheap_upscale(self, image: np.ndarray, interpolation=cv.INTER_LINEAR) -> np.ndarray:
        hight, width = image.shape[:2]
        return cv.resize(image, (width * self.up_factor, hight * self.up_factor), interpolation=interpolation)

    def up_scale(self, image: np.ndarray, H=None) -> np.ndarray:
        """ Use raisr algorithm to upscale the image.
        The raisr algorithm only process the y channel of image's ycrcb mode, the other two channel still use cheap
        upscale method.
        :param H: pre-trained filters
        :param image: Image with BGR mode
        :return: Upscaled image with BGR mode
        """
        if H is not None and self.H is None:
            with open(H, "rb") as f:
                self.H = pickle.load(f)
        img_ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        y: np.ndarray = img_ycrcb[:, :, 0]
        crcb = img_ycrcb[:, :, 1:]

        # standardlize
        std = y.std()
        mean = y.mean()
        h, w = y.shape
        h *= self.up_factor
        w *= self.up_factor
        y_standed = (y - mean) / std

        if self.H is None:
            raise Exception("The model have not learned the filters")

        # cheap upscale and pad
        top_pad = (self.patch_size - 1) // 2
        if self.patch_size % 2 == 0:
            bottom_pad = top_pad + 1
        else:
            bottom_pad = top_pad
        y_standed_upscaled_padded = cv.copyMakeBorder(cv.resize(y_standed, (w, h)),
                                                      top_pad, bottom_pad, top_pad, bottom_pad, cv.BORDER_REPLICATE)

        # fit the HR y channel
        hr_y = np.zeros((h, w))
        patches = extract_patches_2d(y_standed_upscaled_padded, (self.patch_size, self.patch_size))

        def f(item):
            idx, patch = item
            # origin coordinate
            x = idx // w
            y = idx % w

            # compute pixel type
            t = x % self.up_factor * self.up_factor + y % self.up_factor
            # conpute hash key j
            angle, coherence, strength = self.hash_key_gen.gen_hash_key(patch)
            filter1d = self.H[angle, coherence, strength, t]
            hr_y[x, y] = patch.ravel().T.dot(filter1d)

        with mp.Pool(processes=mp.cpu_count()) as ps:
            ps.map(f, enumerate(patches))

        # de-standardlize
        hr_y_destanded = hr_y * std + mean
        # cheap upscale the left two channel
        crcb_upscaled = cv.resize(crcb, (w, h), interpolation=cv.INTER_LINEAR)

        hr_ycrcb = np.dstack((hr_y_destanded.astype(np.uint8), crcb_upscaled))
        return cv.cvtColor(hr_ycrcb, cv.COLOR_YCrCb2BGR)
