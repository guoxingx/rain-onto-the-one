
import numpy as np
from cv2 import cv2 as cv

from utils import get_bucket_power_from_file, log
from params import STATICS_DIR, SIZE, IMAGES_DIR
from generator import load_raw, load_image, save_raw_list


def differential(bucket_list, reference_list, size):
    """
    differential ghost imaging (DGI)

    G = <A(x,y), Y> - (<Y>/<R>) * <A(x,y), R>
        R = sum(A(x,y))

    reference     : A(x,y)
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
    size = SIZE

    bucket_list = get_bucket_power_from_file("0331_SCU_60/output.txt")
    bucket_list = bucket_list[:3600]
    reference_list = load_raw("raw60.txt", size=size, bio=False)
    reference_list = reference_list[:3600]

    dgi = differential(bucket_list, reference_list, size)
    cv.imwrite("{}/result_differential.png".format(STATICS_DIR), dgi)

