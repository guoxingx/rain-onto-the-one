
"""
# random choice [number] elements in [sample], no repeat if set replace=False
numpy.random.choice(sample, number, replace=True)

# convert matrix into an array
ndarray.flat[]

# 伪逆
numpy.linalg.pinv(A)

# 返回每 行/列 中最大元素的位置，axis=0:列，axis=1:行
numpy.argmax(array, axis=0)

# 在a的末尾拼接b的index列
a = np.column_stack((a, b[:,index,True]))

# cv.dct(y) 默认ortho, 与 scipy.fftpack.dct(y, norm='ortho')
"""

import math
import numpy as np
import scipy.fftpack as spfft
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
from cv2 import cv2 as cv

import utils


def omp(A, y, K):
    """
    @params: A: <numpy.ndarray>: 传感矩阵 A = phi * psi
    @params: y: <numpy.ndarray>: 测量值 y = phi * x
    @params: K: <int>          : 稀疏系数

    @return: xr: <numpy.ndarray>
    """
    M = A.shape[0]
    N = A.shape[1]

    # 存储恢复的列向量
    theta = np.zeros((N, 1))

    # 存储迭代过程中A被选择的列 和 列序号
    Ac = np.zeros((M, 0))
    index = []

    # 残差，初始化为y [M, 1]
    r = y
    for i in range(K):
        # 1. 找到r和A列积最大值的对应位置
        product = np.fabs(A.T.dot(r))
        # pos = product.argmax()
        pos = product.sum(1).argmax()
        # print("shape: {}, loop: {}, pos: {}".format(product.shape, i, pos))

        # 将A逐列赋值给Ac，True表示以矩阵形式
        Ac = np.column_stack((Ac, A[:,pos,True]))
        index.append(pos)

        # 清零A的对应列
        # A[:,pos] = np.zeros(M)

        # 最小二乘解 x = [ (AT * A) ^ -1 ] * AT * y
        tmp = np.linalg.pinv(Ac.T.dot(Ac))
        ls = tmp.dot(Ac.T).dot(y)
        r = y - Ac.dot(ls)

    for i, pos in enumerate(index):
        theta[pos] = ls[i]
    return theta


def mock_restruct_1d():
    M = 64
    N = 256
    K = 20

    # # 一维信号 [N, 1]
    # x = np.zeros((N, 1))

    # # 稀疏信号随机模拟
    # Index_K = np.random.choice(N, K, replace=False)
    # x.flat[Index_K] = np.random.randn(K) * 10

    x = np.random.rand(N, 1)

    # 观测矩阵 [M, N]
    phi = np.random.randn(M, N)

    # 稀疏基矩阵 [N, N]
    # psi = np.eye(N)
    psi = utils.dct_1d_matrix(N)
    # psi = cv.idct(np.eye(N))
    # psi = cv.idct(np.eye(N)) / np.sqrt(N)
    # psi = cv.dct(np.eye(N))

    # psi = np.zeros((N, N))
    # v = range(N)
    # for k in range(N):
    #     dct_1d = np.cos(np.dot(v, k*math.pi/N))
    #     if k > 0:
    #         dct_1d = dct_1d - np.mean(dct_1d)
    #     psi[:,k] = dct_1d/np.linalg.norm(dct_1d)

    # 传感矩阵 [M, N]
    A = np.dot(phi, psi)

    # 一维测量值 [M, 1]
    y = np.dot(phi, x)

    theta = omp(A, y, K)
    xr = np.dot(psi, theta)

    # print("x: ")
    # for i, k in enumerate(x):
    #     if k[0] != 0:
    #         print(i, k[0])

    # print("\nxr: ")
    # for i, k in enumerate(xr):
    #     if k[0] != 0:
    #         print(i, k[0])

    plt.plot(x)
    for i, v in enumerate(xr):
        if abs(v) > 0.1:
            plt.scatter(i, v)
    plt.show()


def mock_restruct_image_1d():
    M = 128 * 128
    N = 379 * 379
    K = 30 * 30

    x = cv.imread("statics/xiongji.jpeg", 0)
    x = x.reshape((N, 1))

    # 观测矩阵 [M, N]
    phi = np.random.randn(M, N)

    # 稀疏基矩阵 [N, N]
    psi = np.eye(N)

    # 传感矩阵 [M, N]
    A = phi.dot(psi)

    # 一维测量值 [M, N]
    y = phi.dot(x)

    theta = omp(A, y, K)
    xr = np.dot(psi, theta)
    xr = xr.reshape((379, 379))

    cv.imwrite("statics/omp_M{}_K{}.png".format(M, K), xr)

    # plt.imshow(psi)
    # plt.show()


def mock_restruct_image():
    M = 127
    N = 378
    K = 30

    x = cv.imread("statics/xiongji.jpeg", 0).astype(float)
    x = x[:N,:N]
    dct = utils.dct_1d_matrix(N)
    x = dct.dot(x)

    # 观测矩阵 [M, N]
    phi = np.random.randn(M, N)

    # 稀疏基矩阵 [N, N]
    psi = np.eye(N)
    # psi = utils.dct_1d_matrix(N)
    # psi = utils.dct_matrix(N)

    # 传感矩阵 [M, N]
    A = phi.dot(psi)

    # 一维测量值 [M, N]
    y = phi.dot(x)

    sparse_rec_1d = np.zeros((N, N))
    for i in range(N):
        theta = omp(A, y[:,i,True], K)
        sparse_rec_1d[:,i,True] = theta
    xr = np.dot(psi, sparse_rec_1d)
    xr = np.linalg.pinv(dct).dot(xr)

    cv.imwrite("statics/omp_M{}_K{}.png".format(M, K), xr)

    # plt.imshow(psi)
    # plt.show()


def test():
    N = 40
    psi = utils.dct_1d_matrix(N)
    psi2 = utils.dct_2d_matrix(N, N)
    dct = utils.dct_matrix(N)

    # a = np.array([[5,5,5,5],[4,4,4,4],[3,3,3,3],[5,5,5,5]]).astype(float)
    # print(psi.dot(a).dot(psi.T))
    # print(cv.dct(a))

    b = np.array([15,12,14,17]).astype(float)
    b = np.random.rand(N, 1)
    print(b)
    print(psi.dot(b))
    print(cv.dct(b))
    # plt.plot(b)
    # plt.plot(cv.dct(b))
    # plt.show()


if __name__ == "__main__":
    # test()
    # mock_restruct_1d()
    # mock_restruct_image_1d()
    mock_restruct_image()
