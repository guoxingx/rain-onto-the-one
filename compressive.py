
"""
# 求范数
numpy.linalg.norm(a, ord=None, ...)
    a   : matrix
    ord : 1 - 1范数(最大列和), np.inf - 无穷范数(最大行和)
          2(默认) - 2范数 - (A.T * A)的最大特征值的平方根

"""

import math
import numpy as np
from cv2 import cv2 as cv


NSIZE = 379
SIZE = (NSIZE, NSIZE)
FILEPATH = "statics/xiongji.jpeg"

SampleRate = 0.7


def get_object(filepath=FILEPATH):
    """
    """
    image = cv.imread(filepath, 0)
    print("image.shape: ", image.shape)
    return image


def get_phi_matrix(size=SIZE, sample_rate=SampleRate):
    """
    """
    phi = np.random.randn(int(size[0]*sample_rate), size[1])
    print("phi: {}".format(phi.shape))
    return phi


def get_sparse_matrix(size=SIZE):
    """
    稀疏dct矩阵
    """
    if size[0] != size[1]:
        raise ValueError("invalid size: {}".format(size))

    length = size[0]

    mat_dct_1d = np.zeros(size)
    v = range(size[0])
    for k in range(0, length):
        dct_1d = np.cos(np.dot(v, k*math.pi/length))
        # print("{}: dct_1d: {}".format(k, dct_1d))
        if k > 0:
            # 减去均值
            dct_1d = dct_1d - np.mean(dct_1d)
        # print("{}: dct_1d fix: {}".format(k, dct_1d))

        # 求范数
        mat_dct_1d[:,k] = dct_1d/np.linalg.norm(dct_1d)
        # print("mat_dct_1d update: {}\n".format(mat_dct_1d))

    return mat_dct_1d


def cs_omp(y, A, k):
    """
    y [M]
    """

    # 残差
    residual = y

    # smddx
    # index = np.zeros((l), dtype=int)
    index = np.zeros((1, k))
    print("index.shape: ", index.shape)
    print("A.shape: {}, {}".format(A.shape, A[:,False].shape))

    result = np.zeros(NSIZE)
    for j in range(k):
        # 矩阵转置
        dt = A.T

        # fabs: 绝对值
        product = np.fabs(np.dot(dt, residual))

        # 最大投影系数的对应位置
        pos = np.argmax(product)

        index.flat[j] = pos

        # 最小二乘
        # my = np.linalg.pinv(A[:,index>=0])
        my = np.linalg.pinv(A[:])
        a = np.dot(my, y)
        # print("my: {}".format(my.shape))
        # print("y: {}".format(y.shape))
        # print("a: {}".format(a.shape))

        # residual = y - np.dot(A[:,index>=0], a)
        residual = y - np.dot(A[:], a)
    print("a: {}, result: {}".format(a.shape, result.shape))
    result[index>0]=a
    return result


def restruct():
    # 稀疏基矩阵 (N, N)
    dct = get_sparse_matrix()

    # 测量矩阵 (M, N)
    phi = get_phi_matrix()

    # 物体 (N, N)
    image = get_object("statics/xiongji.jpeg")

    # 测量针 * 待测物 (M, N)
    image_cs_1d = np.dot(phi, image)

    # 测量阵 * 稀疏基矩阵 (M, N)
    A = np.dot(phi, dct)
    # print("A: {}".format(A))

    # 稀疏系数
    sparse_rec_1d = np.zeros((NSIZE, NSIZE))
    for i in range(NSIZE):
        print("正在重建: {}行".format(i))

        column_rec = cs_omp(image_cs_1d[:,i], A, 10)
        sparse_rec_1d[:,i] = column_rec
    image_rec = np.dot(mat_dct_1d, sparse_rec_1d)


if __name__ == "__main__":
    restruct()

    # dct = get_sparse_matrix((6,6))
    # phi = get_phi_matrix((6,6))
