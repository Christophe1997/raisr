import cv2 as cv
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from raisr.utils import is_image, build_goal_func
import os
import pickle
from raisr.hash_key_gen import gen_hash_key
from raisr.processor import UNSHARP_MASKING_5x5_KERNEL, gaussian_kernel_2d
from datetime import datetime
import time
from pathos import multiprocessing as mp
from scipy.optimize import minimize


class RAISR:

    def __init__(self, upscale_factor=2,
                 angle_base=24, strength_base=9, patch_size=9):
        """RAISR model

        :param upscale_factor: upsampling factor;
        :param angle_base: factor for angle
        """
        self.up_factor = upscale_factor
        self.angle_base = angle_base
        self.strength_base = strength_base
        self.patch_size = patch_size
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
        QS_DIR = os.path.join(dst, "Qs")
        VS_DIR = os.path.join(dst, "Vs")
        if not os.path.exists(dst):
            os.mkdir(dst)
        if not os.path.exists(QS_DIR):
            os.mkdir(QS_DIR)
        if not os.path.exists(VS_DIR):
            os.mkdir(VS_DIR)

        d2 = self.patch_size ** 2
        t2 = self.up_factor ** 2

        Q = np.zeros((self.angle_base, self.strength_base, t2, d2, d2))
        V = np.zeros((self.angle_base, self.strength_base, t2, d2, 1))
        H = np.zeros((self.angle_base, self.strength_base, t2, d2, 1))

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

        def f(item):
            idx, lr_fname = item
            print("*****START TO PROCESS {}/{} IMAGE PAIR AT {}*****".format(
                idx + 1, total, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            hr_fname = hr_filrs[idx]
            lr = cv.imread(os.path.join(lr_path, lr_fname))
            hr = cv.imread(os.path.join(hr_path, hr_fname))

            h, w = hr.shape[:2]
            lr_upscaled = cv.resize(lr, (w, h))
            # Extrat the luminance in YCrCb mode of a image
            lr_y = cv.cvtColor(lr_upscaled, cv.COLOR_BGR2YCrCb)[:, :, 0]
            hr_y = cv.cvtColor(hr, cv.COLOR_BGR2YCrCb)[:, :, 0]

            # normalize
            lr_y = (lr_y - lr_y.min()) / (lr_y.max() - lr_y.min())
            # Pad the image
            lr_y = cv.copyMakeBorder(lr_y, top_pad, bottom_pad, top_pad, bottom_pad, borderType=cv.BORDER_REPLICATE)
            # optionally sharpen
            if sharpen:
                hr_y = cv.filter2D(hr_y, -1, UNSHARP_MASKING_5x5_KERNEL, borderType=cv.BORDER_REPLICATE)

            hr_y = (hr_y - hr_y.min()) / (hr_y.max() - hr_y.min())

            patches = extract_patches_2d(lr_y, (self.patch_size, self.patch_size))

            for idx1, patch in enumerate(patches):
                # origin coordinate
                x = idx1 // w
                y = idx1 % w

                # compute pixel type
                t = x % self.up_factor * self.up_factor + y % self.up_factor
                # conpute hash key j
                angle, strength = gen_hash_key(patch, self.angle_base, self.strength_base)
                patch: np.ndarray
                # compute p_k
                a = patch.ravel().reshape((1, -1))
                # compute x_k, the true HR pixel
                b = hr_y[x, y]

                a_T_a = a.T.dot(a)
                a_T_b = a.T.dot(b)

                # increase date by flip and rotate
                rot90 = self.angle_base // 4
                for i in range(4):
                    angle1 = (angle + i * rot90) % self.angle_base
                    Q[angle1, strength, t] += a_T_a
                    V[angle1, strength, t] += a_T_b
                    a_T_a = np.rot90(a_T_a)
                a_T_a = np.flipud(a_T_a)
                flipud = self.angle_base - angle
                for i in range(4):
                    angle1 = (flipud + i * rot90) % self.angle_base
                    Q[angle1, strength, t] += a_T_a
                    V[angle1, strength, t] += a_T_b
                    a_T_a = np.rot90(a_T_a)

            with open(f"{QS_DIR}/Q_{lr_fname}.dat", "wb") as Q_f:
                pickle.dump(Q, Q_f)

            with open(f"{VS_DIR}/V_{lr_fname}.dat", "wb") as V_f:
                pickle.dump(V, V_f)
            print("*****END   TO PROCESS {}/{} IMAGE PAIR AT {}*****".format(
                idx + 1, total, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        with mp.Pool() as ps:
            ps.map(f, enumerate(lr_files))

        print(f"*****START TO SOLVE THE OPTIMIZATION PROBLEM AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*****")
        for Q_f in os.listdir(f"{QS_DIR}"):
            with open(f"{QS_DIR}/{Q_f}", "rb") as f:
                Q += pickle.load(f)
        for V_f in os.listdir(f"{VS_DIR}"):
            with open(f"{VS_DIR}/{V_f}", "rb") as f:
                V += pickle.load(f)
        # compute H
        for angle in range(self.angle_base):
                for strength in range(self.strength_base):
                    for t in range(t2):
                        # solve the optimization problem by a conjugate gradient solver
                        goal_func = build_goal_func(Q[angle, strength, t],
                                                    V[angle, strength, t])
                        res = minimize(goal_func, np.random.random((d2, 1)), method='BFGS')
                        H[angle, strength, t] = res.x[:, np.newaxis]
        print(f"*****END   TO SOLVE THE OPTIMIZATION PROBLEM AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*****\n")
        # write the filter
        end = datetime.now()
        timestamp = time.mktime(end.timetuple())

        dump_path = os.path.join(dst, "raisr_filter_{}x{}_{}x{}_{:.0f}.pkl".format(
            self.patch_size, self.patch_size,
            self.angle_base, t2,
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

        h, w = image.shape[:2]
        h *= self.up_factor
        w *= self.up_factor

        img_upscaled = cv.resize(image, (w, h))
        if self.H is None:
            raise Exception("The model have not learned the filters")

        # cheap upscale and pad
        top_pad = (self.patch_size - 1) // 2
        if self.patch_size % 2 == 0:
            bottom_pad = top_pad + 1
        else:
            bottom_pad = top_pad

        img_upscaled_ycrcb = cv.cvtColor(img_upscaled, cv.COLOR_BGR2YCrCb)
        y = img_upscaled_ycrcb[:, :, 0]
        crcb = img_upscaled_ycrcb[:, :, 1:]

        # normalize
        m = y.max()
        n = y.min()
        y = (y - n) / (m - n)
        y_padded = cv.copyMakeBorder(y, top_pad, bottom_pad, top_pad, bottom_pad, cv.BORDER_REPLICATE)

        start = datetime.now()
        # fit the HR y channel
        patches = extract_patches_2d(y_padded, (self.patch_size, self.patch_size))

        def f(item):
            idx, patch = item
            x = idx // w
            y = idx % w

            # compute pixel type
            t = x % self.up_factor * self.up_factor + y % self.up_factor
            # conpute hash key j
            angle, strength = gen_hash_key(patch, self.angle_base, self.strength_base)
            filter1d = self.H[angle, strength, t]
            return patch.ravel().T.dot(filter1d)

        with mp.Pool() as ps:
            ret = ps.map(f, enumerate(patches))

        hr_y = np.array(ret).reshape((h, w))
        # de-normalize
        hr_y = hr_y * (m - n) + n
        # cheap upscale the left two channel
        hr_ycrcb = np.dstack((hr_y.astype(np.uint8), crcb))
        end = datetime.now()
        print(f"*****FINISH UPSCALE, TOTAL COSTS {(end - start).total_seconds()}s*****")
        return cv.cvtColor(hr_ycrcb, cv.COLOR_YCrCb2BGR)
