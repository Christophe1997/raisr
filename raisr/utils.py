import os


def is_image(fp_path: str):
    return fp_path.lower().endswith(('.png', '.jpg', '.tiff', '.bmp', '.gif'))
