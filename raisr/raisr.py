import cv2 as cv
import numpy as np
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

        self.hash_key_gen = HashKeyGen(angle_base, strength_base, coherence_base)

    def __repr__(self):
        return f"<RAISR up_factor={self.up_factor}>"

    def train(self, hr, lr, batch, max_step):
        pass

    def cheap_upscale(self, image: np.ndarray, interpolation=cv.INTER_LINEAR) -> np.ndarray:
        hight, width = image.shape[:2]
        return cv.resize(image, (width * self.up_factor, hight * self.up_factor), interpolation=interpolation)
