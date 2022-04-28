
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
"""

import math
import numpy as np
import scipy.fftpack as spfft
import matplotlib.pyplot as plt

M = 64
N = 256
K = 5

# 一维信号 [N, 1]
x = np.zeros((N, 1))

# 稀疏信号随机模拟
Index_K = np.random.choice(N, K, replace=False)
x.flat[Index_K] = np.random.randn(K) * 5

# x.flat[[1,45,142,191,249]] = [1,2,3,4,5]

# 观测矩阵 [M, N]
phi = np.random.randn(M, N)

# 稀疏基矩阵 [N, N]
psi = np.identity(N)

# 传感矩阵 [M, N]
A = np.dot(phi, psi)

# 一维测量值 [M, 1]
y = np.dot(phi, x)

# 存储恢复的列向量
theta = np.zeros((N, 1))

# 存储迭代过程中A被选择的列 和 列序号
# Ac = np.zeros((M, K))
Ac = np.zeros((M, 0))
index = []

# 残差，初始化为y [M, 1]
r = y

# for i in range(K*2):
for i in range(K):
    # 1. 找到r和A列积最大值的对应位置
    product = np.dot(A.T, r)
    pos = np.argmax(np.fabs(product))
    print("loop: {}, pos: {}".format(i, pos))

    # 将A逐列赋值给Ac，True表示以矩阵形式
    Ac = np.column_stack((Ac, A[:,pos,True]))
    index.append(pos)

    # 清零A的对应列
    A[:,pos] = np.zeros(M)

    # 最小二乘解 x = [ (AT * A) ^ -1 ] * AT * y
    tmp = np.linalg.pinv(Ac.T.dot(Ac))
    ls = tmp.dot(Ac.T).dot(y)

    r = y - Ac.dot(ls)


for i, pos in enumerate(index):
    theta[pos] = ls[i]
x_r = np.dot(psi, theta)

print("x: ")
for i, k in enumerate(x):
    if k[0] != 0:
        print(i, k[0])

print("\nx_r: ")
for i, k in enumerate(x_r):
    if k[0] != 0:
        print(i, k[0])
# print("x_r: {}".format(x_r))

# plt.plot(x)
# plt.plot(x_r)
# plt.show()
