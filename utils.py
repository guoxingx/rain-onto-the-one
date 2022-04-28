
import time
from datetime import datetime

from cv2 import cv2 as cv
import scipy.fftpack as spfft


def log(s):
    now = datetime.now()
    print("{}: {}".format(now, s))


def convert_grayscale_image(path, save=False):
    """
    save a bio image from a existed image with bio_filename
    """
    res = cv.imread(path, 0)
    if save:
        bio_path = "{}/{}{}".format(STATICS_DIR, "gray_", path)
        cv.imwrite(bio_path, res)
    return res


def get_bucket_power_from_file(filepath):
    """
    load processed output file of power meter
    """
    bs = []
    f = open(filepath, 'r')
    for i, line in enumerate(f.readlines()):
        bs.append(float(line))
    f.close()
    return bs


def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
