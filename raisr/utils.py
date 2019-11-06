import os


def is_image(fp_path: str):
    if os.path.exists(fp_path):
        return fp_path.lower().endswith(('.png', '.jpg', '.tiff', '.bmp', '.gif'))
    else:
        return False
