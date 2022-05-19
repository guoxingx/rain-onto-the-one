
"""
# 函数返回的是数组值从小到大的索引值, axis=-1, 0, 1, None, ...
numpy.argsort(a, axis=-1, ...)
"""
import numpy as np
import matplotlib.pyplot as plt


def cosamp(A, y, K):
    """
    @params: A: <numpy.ndarray>: 传感矩阵 A = phi * psi
    @params: y: <numpy.ndarray>: 测量值 y = phi * x
    @params: K: <int>          : 稀疏系数

    @return: xr: <numpy.ndarray>
    """
    (M, N) = A.shape

    # 存储恢复的列向量
    # theta = np.zeros((N, 1))
    theta_pos = np.array([])

    # 残差，初始化为y [M, 1]
    r = y

    for i in range(K):
        # # A的各列与残差r 内积模最大的2K列
        # product_pos = np.fabs(A.T.dot(r)).argsort(None)[::-1][:2*K]
        # Is = np.union1d(product_pos, theta_pos).astype(int)
        # At = A[:,product_pos]

        # # 最小二乘解 ls = [ (AT * A) ^ -1 ] * AT * y (2K, 1)
        # ls = np.linalg.pinv(At.T.dot(At)).dot(At.T).dot(y)

        # ls_pos = np.fabs(ls).argsort(None)[::-1][:K]
        # lsk = ls[ls_pos]
        # theta_pos = Is[ls_pos]

        # r = y - At[:,ls_pos].dot(lsk)

        product_pos = np.fabs(A.T.dot(r)).argsort(0)[::-1][:2*K]
        theta_pos = np.union1d(product_pos, theta_pos).astype(int)

        theta = np.zeros((N,1))
        # ls = np.dot(np.linalg.pinv(A[:,theta_pos]), y)
        At = A[:,theta_pos]
        ls = np.linalg.pinv(At.T.dot(At)).dot(At.T).dot(y)

        theta[theta_pos] = ls
        theta_pos = np.fabs(theta).argsort(0)[::-1][:K]
        r = y - A.dot(theta)
    return theta


if __name__ == "__main__":
    M = 128
    N = 256
    K = 20

    # 一维信号 [N, 1]
    x = np.zeros((N, 1))

    # 稀疏信号随机模拟
    Index_K = np.random.choice(N, K, replace=False)
    x.flat[Index_K] = np.random.randn(K) * 10

    # 观测矩阵 [M, N]
    phi = np.random.randn(M, N)

    # 稀疏基矩阵 [N, N]
    psi = np.eye(N)

    # 传感矩阵 [M, N]
    A = np.dot(phi, psi)

    # 一维测量值 [M, 1]
    y = np.dot(phi, x)

    theta = cosamp(A, y, K)
    xr = np.dot(psi, theta)
    # theta, _ = cs_CoSaMP(y, A, K)
    # xr = psi.dot(theta)

    for i in range(N):
        if abs(x[i]) > 0.1:
            print("x : {}: {}".format(i, x[i]))
        if abs(xr[i]) > 0.1:
            print("xr: {}: {}".format(i, xr[i]))

    plt.plot(x)
    for i, v in enumerate(xr):
        if abs(v) > 0.1:
            plt.scatter(i, v)
    plt.show()
