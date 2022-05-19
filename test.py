import numpy as np
import matplotlib.pyplot as plt


def cs_CoSaMP(y,Phi,K):
    residual=y  #初始化残差
    (M,N) = Phi.shape
    index = np.array([])
    result=np.zeros((N,1))
    for j in range(K):  #迭代次数
        product=np.fabs(np.dot(Phi.T,residual))         # 计算投影
        top_k_idx = product.argsort(axis=0)[-2*K:]        # 取最大的K个的序号
        index = np.union1d(index,top_k_idx).astype(int) # 更新候选集
        x = np.zeros((N,1))                             # 算一部分x
        x_temp = np.dot(np.linalg.pinv(Phi[:,index]),y) # 最小二乘
        x[index] = x_temp                               # 放回去
        index = np.fabs(x).argsort(axis=0)[-K:]         # 取最大的K个的序号
        residual=y-np.dot(Phi,x)                        # 更新残差
    return  x, index


# 单次实验
N = 256
M = 128
K = 20
# 生成稀疏信号（高斯）
x = np.random.randn(N,1)
x[:N-K]=0
np.random.shuffle(x)
# 生成高斯随机测量矩阵
Phi=np.random.randn(M,N)/np.sqrt(M)
psi = np.eye(N)
# 观测信号
y = np.dot(Phi,x)
A = Phi.dot(psi)
theta, _ = cs_CoSaMP(y,A,K)
xr = psi.dot(theta)
# print(Candidate)
error = np.linalg.norm(x-xr)/np.linalg.norm(x)
print(error)

for i in range(N):
    if abs(x[i]) > 0.1:
        print("x : {}: {}".format(i, x[i]))
    if abs(xr[i]) > 0.1:
        print("xr: {}: {}".format(i, xr[i]))
