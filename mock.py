
import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt

from conventional import conventional
from hadamard import generate_speckle


def gi():
    """
    """
    M, N = 128, 128
    size = (N, N)

    x = cv.imread("statics/xiongji.jpeg", 0).astype(float)
    # x = cv.imread("statics/lena.jpg", 0).astype(float)
    # x = cv.imread("data/0713_13_1_H32_W532/convential_nor.png", 0).astype(float)
    x = cv.resize(x, size)

    rlist_h = generate_speckle(N)
    rlist_r = generate_speckle_random(N)

    blist_h = []
    blist_r = []
    for i in range(N*N*2):
        if i < N*N:
            # bh = (rlist_h[i].dot(x)).sum()
            bh = (rlist_h[i] * x).sum()
            blist_h.append(bh)

        # br = (rlist_r[i].dot(x)).sum()
        br = (rlist_r[i] * x).sum()
        blist_r.append(br)

    gih = conventional(blist_h, rlist_h, size)
    cv.imwrite("statics/xiongji_h{}.jpeg".format(N), gih)

    gir = conventional(blist_r, rlist_r, size)
    cv.imwrite("statics/xiongji_r{}.jpeg".format(N), gir)

    plt.subplot(2,2,1)
    plt.title("GI_Hadamard")
    plt.imshow(gih)

    plt.subplot(2,2,2)
    plt.title("GI_Random")
    plt.imshow(gir)

    plt.show()


def generate_speckle_random(N=40):
    nds = []
    for index in range(N*N*2):
        nd = np.random.binomial(1, 0.5, (N, N))
        nd = nd * 255
        nds.append(nd)
    return nds


if __name__ == "__main__":
    gi()

