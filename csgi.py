
import math
import enum

import numpy as np
from cv2 import cv2 as cv

import utils
from params import STATICS_DIR, SIZE, IMAGES_DIR
from generator import load_raw, save_raw_list
from omp import omp
from cosamp import cosamp


class CSAlg(enum.Enum):
    omp = 1
    cosamp = 2


def csgi(y, phi, K, typ=CSAlg.omp):
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

    if typ == CSAlg.omp:
        theta = omp(A, y, K)
    elif typ == CSAlg.cosamp:
        theta = cosamp(A, y, K)
    else:
        return TypeError("unrecognized or unimplemented compressive algorithm.")
    xr = np.dot(psi, theta)

    mi = np.min(xr)
    mx = np.max(xr)
    gi = 255 * (xr - mi) / (mx - mi)
    return gi


if __name__ == "__main__":
    size = (40,40)
    filedir = "data/outdated/0628_16_1/"

    N = 1600
    M = 800
    K = 200
    typ = "omp"

    bucket_list = utils.get_bucket_power_from_file("{}/output.txt".format(filedir))
    psi = utils.dct_1d_matrix(N)
    x = np.array(bucket_list[:N])
    reference_list = load_raw("raw40.txt", size=size, bio=False)
    bucket_list = bucket_list[:M]
    reference_list = reference_list[:M]

    y = np.array(bucket_list).reshape((M,1))
    phi = np.zeros((M, N))
    for i in range(M):
        phi[i]= np.array(reference_list[i]).reshape(N)

    gi = csgi(y, phi, K)
    gi = gi.reshape(size)
    cv.imwrite("{}/cs_{}_M{}_K{}.png".format(filedir, typ, M, K), gi)
