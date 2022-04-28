#!/usr/bin/env python


from cv2 import cv2 as cv

from params import SIZE
from generator import load_raw
from conventional import conventional
from differential import differential
from utils import get_bucket_power_from_file


if __name__ == "__main__":
    dir_str = input(
        """
        dir of restructed data: ("data" as default)
        """
    )
    dirname = dir_str.strip()

    size = SIZE

    bucket_list = get_bucket_power_from_file("{}/output.txt".format(dirname))
    bucket_list = bucket_list[:3600]
    reference_list = load_raw("raw60.txt", size=size, bio=False)
    reference_list = reference_list[:3600]

    gi_c = conventional(bucket_list, reference_list, size)
    cv.imwrite("{}/convential.png".format(dirname), gi_c)

    gi_d = differential(bucket_list, reference_list, size)
    cv.imwrite("{}/differential.png".format(dirname), gi_d)
