
import time
from datetime import datetime

from cv2 import cv2 as cv


def log(s):
    now = datetime.now()
    print("{}: {}".format(now, s))


def convert_bio_image(path, save=False):
    res = cv.imread(path, 0)
    if save:
        bio_path = "{}/{}{}".format(STATICS_DIR, "bio_", path)
        cv.imwrite(bio_path, res)
    return res


def get_bucket_power_from_file(filepath):
    tss = []
    f = open(filepath, 'r')
    for i, line in enumerate(f.readlines()):
        ts = float(line)
        if ts > 1000000000000:
            ts = ts / 1000
        tss.append(ts)
    f.close()
    return tss


