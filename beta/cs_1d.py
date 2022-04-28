"""
# 创建变量
cvxpy.Variable(shape=())

# 求范数
cvxpy.norm(x, p=2)

# 约束
constraints = [ ? == ? ]

# 删除多余维度
numpy.squeeze()

# 生成n*n的单位矩阵
numpy.identity(n)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cp


n = 5000
t = np.linspace(0, 1/8, n)
y = np.sin(1394 * np.pi * t) + np.sin(3266 * np.pi * t)
yt = spfft.dct(y, norm='ortho')

m = 500 # 10% sample
ri = np.random.choice(n, m, replace=False) # random sample of indices
ri.sort() # sorting not strictly necessary, but convenient for plotting
t2 = t[ri]
y2 = y[ri]


# create idct matrix operator
# A 是单位阵在时间域10%取样的矩阵
#
# 单位阵做 idct
A = spfft.idct(np.identity(n), norm='ortho', axis=0)

# 选出 10% 的信号
A = A[ri]


# do L1 optimization
vx = cp.Variable(n)

# A
# 找到最小的1范数
objective = cp.Minimize(cp.norm(vx, 1))

# 条件：A*x = y
constraints = [A*vx == y2]

prob = cp.Problem(objective, constraints)
result = prob.solve(verbose=True)
print("result: {}".format(result))
print("x: {}".format(vx.value))


# reconstruct signal
x = np.array(vx.value)
x = np.squeeze(x)
sig = spfft.idct(x, norm='ortho', axis=0)


# plt.plot(t, y)
# plt.plot(1/t, yt)
# plt.plot(t2, y2)
plt.plot(t, sig)
plt.plot(1/t, x)
plt.show()
