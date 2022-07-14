
import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
from scipy import linalg

import hadamard
from conventional import conventional
from omp import omp

import utils


def gi():
    """
    """
    M, N = 64, 64
    size = (N, N)

    # x = cv.imread("statics/xiongji.jpeg", 0).astype(float)
    x = cv.imread("statics/lena.jpg", 0).astype(float)
    # x = cv.imread("data/0713_13_1_H32_W532/convential_nor.png", 0).astype(float)
    x = cv.resize(x, size)

    rlist_h = hadamard.generate_speckle(N)
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
    # cv.imwrite("statics/xiongji_h{}.jpeg".format(N), gih)

    gir = conventional(blist_r, rlist_r, size)
    # cv.imwrite("statics/xiongji_r{}.jpeg".format(N), gir)

    plt.subplot(2,2,1)
    plt.title("GI_Hadamard")
    plt.imshow(gih)

    plt.subplot(2,2,2)
    plt.title("GI_Random")
    plt.imshow(gir)

    plt.show()


def csgi():
    M = 1024
    K = 256
    n = 64
    N = n * n

    size = (n, n)
    x = cv.imread("statics/xiongji.jpeg", 0).astype(float)
    x = cv.resize(x, size)
    x = x.reshape((N, 1))

    # 观测矩阵 [M, N]
    # phi = np.random.randn(M, N)
    phi = linalg.hadamard(N)[np.random.choice(N, M, replace=False)]

    # 稀疏基矩阵 [N, N]
    psi = utils.dct_1d_matrix(N)

    # 传感矩阵 [M, N]
    A = phi.dot(psi)

    # 一维测量值 [M, N]
    y = phi.dot(x)

    theta = omp(A, y, K)
    xr = np.dot(psi, theta)
    xr = xr.reshape(size)

    # cv.imwrite("statics/omp_M{}_K{}.png".format(M, K), xr)

    # mi = np.min(xr)
    # mx = np.max(xr)
    # gi = 255 * (xr - mi) / (mx - mi)

    plt.imshow(xr)
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
