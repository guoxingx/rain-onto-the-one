
import numpy as np
from cv2 import cv2 as cv

from utils import get_bucket_power_from_file, log, normalize
from params import STATICS_DIR, SIZE, IMAGES_DIR
from generator import load_raw, save_raw_list


def conventional(bucket_list, reference_list, size):
    """
    ghost imaging (GI)

    G = <(B - <B>) * Ir>
      = <B * Ir> - <<B> * Ir>
      = <B * Ir> - <B> <Ir>
    """

    if len(bucket_list) != len(reference_list):
        raise ValueError("bucket counts must be equal to reference counts, bucket: {}, reference: {}"
               .format(len(bucket_list), len(reference_list)))
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
    # log("gi is {}".format(gi))
    return gi


if __name__ == "__main__":
    size = (40, 40)
    filedir = "data/0608_23_D_1_M4"

    b_list = get_bucket_power_from_file("{}/output.txt".format(filedir))
    r_list = load_raw("raw40.txt", size=size, bio=False)

    bucket_list, reference_list = b_list[:], r_list[:]
    bucket_list = normalize(bucket_list)

    gi = conventional(bucket_list, reference_list, size)
    cv.imwrite("{}/convential.png".format(filedir), gi)
