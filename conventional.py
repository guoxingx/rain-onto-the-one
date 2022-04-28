
import numpy as np
from cv2 import cv2 as cv

from utils import get_bucket_power_from_file, log
from params import STATICS_DIR, SIZE, IMAGES_DIR
from generator import load_raw, load_image, save_raw_list


def conventional(bucket_list, reference_list, size):
    """
    ghost imaging (GI)

    G = <(B - <B>) * Ir>
      = <B * Ir> - <<B> * Ir>
      = <B * Ir> - <B> <Ir>
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


if __name__ == "__main__":
    size = SIZE

    bucket_list = get_bucket_power_from_file("0331_SCU_60/output.txt")
    bucket_list = bucket_list[:3600]
    reference_list = load_raw("raw60.txt", size=size, bio=False)
    reference_list = reference_list[:3600]

    gi = conventional(bucket_list, reference_list, size)
    cv.imwrite("{}/convential.png".format(STATICS_DIR), gi)
