
import numpy as np
from cv2 import cv2 as cv

from utils import get_bucket_power_from_file, log, normalize
from params import STATICS_DIR, SIZE, IMAGES_DIR
from generator import load_raw, save_raw_list


def differential(bucket_list, reference_list, size):
    """
    differential ghost imaging (DGI)

    G = <A(x,y), B> - (<B>/<R>) * <A(x,y), R>

        R = sum(A(x,y))

    reference     : A(x,y)
    bucket        : B
    ref_value_sum : R

    """

    if len(bucket_list) != len(reference_list):
        raise ValueError("bucket counts must be equal to reference counts")
    times = len(bucket_list)

    ref_bucket_list = normalize([np.sum(v) for v in reference_list])

    correlate_sum = np.zeros(size)
    bucket_sum = 0
    ref_bucket_sum = 0
    correlate_dif_sum = np.zeros(size)

    for i in range(times):
        reference = reference_list[i]
        bucket = bucket_list[i]

        # <B>
        bucket_sum += bucket

        # <R>
        # ref_bucket_sum += np.sum(reference)
        ref_bucket_sum += ref_bucket_list[i]

        # <A(x,y), B>
        correlate_sum = correlate_sum + reference * bucket

        # <A(x,y), R>
        # correlate_dif_sum = correlate_dif_sum + sum(reference) * reference
        correlate_dif_sum = correlate_dif_sum + ref_bucket_list[i] * reference

    bucket_avg = bucket_sum / times
    ref_bucket_avg = ref_bucket_sum / times
    correlate_avg = correlate_sum / times
    correlate_dif_avg = correlate_dif_sum / times

    dgi = correlate_avg - correlate_dif_avg * bucket_avg / ref_bucket_avg
    # dgi = correlate_avg - (bucket_avg / ref_bucket_avg) * correlate_dif_avg

    mi = np.min(dgi)
    mx = np.max(dgi)
    dgi = 255 * (dgi - mi) / (mx - mi)
    # log("dgi is {}".format(dgi))
    return dgi


if __name__ == "__main__":
    size = (40, 40)
    filedir = "data/0601_10_B_1/"

    bucket_list = get_bucket_power_from_file("{}/output.txt".format(filedir))
    bucket_list = normalize(bucket_list)
    reference_list = load_raw("raw40.txt", size=size, bio=False)

    dgi = differential(bucket_list[:], reference_list[:], size)
    cv.imwrite("{}/differential.png".format(filedir), dgi)
