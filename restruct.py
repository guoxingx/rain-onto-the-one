
import numpy as np
from cv2 import cv2 as cv

from utils import get_bucket_power_from_file, log
from params import STATICS_DIR, SIZE, IMAGES_DIR
from generator import load_raw


def conventional(bucket_list, reference_list, size):
    """
    ghost imaging (GI)

    G = <A(x,y), Y> - <A(x,y)> * <Y>

    reference      : A(x,y)
    bucket        : Y
    """

    if len(bucket_list) != len(reference_list):
        raise ValueError("bucket counts must be equal to reference counts")
    times = len(bucket_list)

    correlate_sum = np.zeros(size)
    reference_sum = np.zeros(size)
    bucket_sum = 0

    for i in range(times):
        bucket = bucket_list[i]
        reference = reference_list[i]

        reference_sum = reference_sum + reference
        bucket_sum += bucket
        correlate_sum = correlate_sum + reference * bucket

    bucket_avg = bucket_sum / times
    reference_avg = reference_sum / times
    correlate_avg = correlate_sum / times

    gi = correlate_avg - bucket_avg * reference_avg
    mi = np.min(gi)
    mx = np.max(gi)
    gi = 255 * (gi - mi) / (mx - mi)
    log("gi is {}".format(gi))
    return gi


def differential(bucket_list, reference_list, size):
    """
    differential ghost imaging (DGI)

    G = <A(x,y), Y> - (<Y>/<R>) * <A(x,y), R>
        R = sum(A(x,y))

    reference      : A(x,y)
    bucket        : Y
    ref_value_sum : R

    """

    if len(bucket_list) != len(reference_list):
        raise ValueError("bucket counts must be equal to reference counts")
    times = len(bucket_list)

    correlate_sum = np.zeros(size)
    reference_sum = np.zeros(size)
    bucket_sum = 0

    ref_bucket_sum = 0
    correlate_dif_sum = np.zeros(size)

    for i in range(times):
        reference = reference_list[i]
        bucket = bucket_list[i]

        reference_sum = reference_sum + reference
        bucket_sum += bucket
        correlate_sum = correlate_sum + reference * bucket

        ref_bucket = np.sum(reference)
        ref_bucket_sum += ref_bucket
        correlate_dif_sum = correlate_dif_sum + reference * ref_bucket

    bucket_avg = bucket_sum / times
    reference_avg = reference_sum / times
    correlate_avg = correlate_sum / times

    ref_bucket_avg = ref_bucket_sum / times
    correlate_dif_avg = correlate_dif_sum / times

    dgi = correlate_avg - (bucket_avg / ref_bucket_avg) * correlate_dif_avg
    mi = np.min(dgi)
    mx = np.max(dgi)
    dgi = 255 * (dgi - mi) / (mx - mi)
    log("dgi is {}".format(dgi))
    return dgi


if __name__ == "__main__":
    size = (40,40)
    bucket_list = get_bucket_power_from_file("analyser/output.txt")
    reference_list = load_raw("raw40.txt", size=size, bio=False)

    gi = conventional(bucket_list, reference_list, size)
    cv.imwrite("{}/result_convential.png".format(STATICS_DIR), gi)

    dgi = differential(bucket_list, reference_list, size)
    cv.imwrite("{}/result_differential.png".format(STATICS_DIR), dgi)

