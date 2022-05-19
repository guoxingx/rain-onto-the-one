
import math

import numpy as np
from cv2 import cv2 as cv

import utils
from params import STATICS_DIR, SIZE, IMAGES_DIR
from generator import load_raw, load_image, save_raw_list
from omp import omp
from cosamp import cosamp


def csgi(y, phi, K, size):
    """
    ghost imaging (GI)

    G = <(B - <B>) * Ir>
      = <B * Ir> - <<B> * Ir>
      = <B * Ir> - <B> <Ir>


    min || dct(T) ||1  s.t.  <Ir * T> = B
    min || x || 1  s.t.  <A * x> = y

    y = B
    A = Ir
    T = x
    """
    M = y.shape[0]
    N = phi.shape[1]

    psi = utils.dct_1d_matrix(N)

    A = phi.dot(psi)

    theta = omp(A, y, K)
    # theta = cosamp(A, y, K)
    xr = np.dot(psi, theta)
    gi = xr.reshape(size)

    mi = np.min(gi)
    mx = np.max(gi)
    gi = 255 * (gi - mi) / (mx - mi)
    return gi


if __name__ == "__main__":
    size = SIZE
    filedir = "0422_SCU_60"

    N = 3600
    M = 3600
    K = 400
    typ = "omp"

    bucket_list = utils.get_bucket_power_from_file("{}/output.txt".format(filedir), N=M)
    reference_list = load_raw("raw60.txt", size=size, bio=False, N=M)

    y = np.array(bucket_list).reshape((M,1))
    phi = np.zeros((M, N))
    for i in range(M):
        phi[i]= np.array(reference_list[i]).reshape(N)

    gi = csgi(y, phi, K, size)
    cv.imwrite("{}/cs_{}_M{}_K{}.png".format(filedir, typ, M, K), gi)
