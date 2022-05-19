
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def cs_omp(y,A,K):
    # 初始化残差
    residual=y
    (M,N) = A.shape
    index=np.zeros(N,dtype=int)
    for i in range(N): # 第i列被选中就是1，未选中就是-1
        index[i]= -1
    result=np.zeros((N,1))
    for j in range(K):  # 迭代次数
        product=np.fabs(np.dot(A.T,residual))
        pos=np.argmax(product)  # 最大投影系数对应的位置
        index[pos]=1 # 对应的位置取1
        my=np.linalg.pinv(A[:,index>=0]) # 最小二乘
        a=np.dot(my,y) # 最小二乘,看参考文献1
        residual=y-np.dot(A[:,index>=0],a)
    result[index>=0]=a
    Candidate = np.where(index>=0) # 返回所有选中的列
    return  result, Candidate


# 单次实验
N = 256
M = 128
K = 10
# 生成稀疏信号（高斯）
x = np.random.randn(N,1)
x[:N-K]=0
np.random.shuffle(x)

print("原信号：")
for i in range(N):
    if abs(x.flat[i]) > 0:
        print("{}: {}".format(i, x.flat[i]))

# 生成高斯随机测量矩阵
Phi=np.random.randn(M,N)/np.sqrt(M)
Psi = np.eye(N)
A = Phi.dot(Psi)
# 观测信号
y = np.dot(A,x)
x_pre, Candidate = cs_omp(y,A,K)
# print(Candidate)
print("\n恢复信号：")
for i, v in enumerate(Candidate[0]):
    print("{}: {}".format(v, x_pre[v]))

error = np.linalg.norm(x-x_pre)/np.linalg.norm(x)
print("误差: ", error)


#读取图像，并变成numpy类型的 array
im = Image.open('statics/lena.jpg').convert('L') #图片大小256*256
# im = Image.open('statics/xiongji.jpeg').convert('L') #图片大小256*256
plt.imshow(im,cmap='gray')
N = 256
M = 128
K = 25

#生成稀疏基DCT矩阵
psi=np.zeros((N,N))
v=range(N)
for k in range(0,N):
    dct_1d=np.cos(np.dot(v,k*math.pi/N))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    psi[:,k]=dct_1d/np.linalg.norm(dct_1d)

im = np.array(im)
# 观测矩阵
Phi=np.random.randn(M,N)/np.sqrt(M)
# 随机测量
img_cs_1d=np.dot(Phi,im)

theta=np.zeros((N,N))   # 初始化稀疏系数矩阵
Theta_1d=np.dot(Phi,psi)   #测量矩阵乘上基矩阵
for i in range(N):
    if i%32==0:
        print('正在重建第',i,'列。')
    y=np.reshape(img_cs_1d[:,i],(M,1))
    column_rec, Candidate=cs_omp(y,Theta_1d,K) #利用OMP算法计算稀疏系数
    x_pre = np.reshape(column_rec,(N))
    theta[:,i]=x_pre


img_rec=np.dot(psi,theta)          #稀疏系数乘上基矩阵
#显示重建后的图片
img_pre=Image.fromarray(img_rec)
plt.imshow(img_pre,cmap='gray')
error = np.linalg.norm(img_rec-im)/np.linalg.norm(im)
plt.show()
