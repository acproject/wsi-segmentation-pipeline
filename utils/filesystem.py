import os
import shutil
import numpy as np


def make_folder(pth, purge=False):
    if purge and os.path.exists(pth):
        shutil.rmtree(pth)
    os.makedirs(pth, exist_ok=True)


def fetch_metadata(pth):
    if os.path.exists(pth):
        return np.load(pth, allow_pickle=True).flatten()[0]
    return {}
