"""

# 放缩图片
scipy.ndimage.zoom(origin, scale)

# read image in grayscale
imageio.imread("path to image", as_gray=True)

"""

import imageio
import numpy as np
import cvxpy as cp
import scipy.ndimage as spimg
import scipy.fftpack as spfft
import matplotlib.pyplot as plt
from pylbfgs import owlqn


from utils import idct2

def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[ri].reshape(b.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx


# read original image and downsize for speed
# read in grayscale
# Xorig = spimg.imread('statics/xiongji.jpeg', flatten=True, mode='L')
Xorig = imageio.imread('statics/xiongji.jpeg', as_gray=True)

sample_sizes = (0.1, 0.01)

# for each sample size
Z = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
masks = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
for i,s in enumerate(sample_sizes):

    # create random sampling index vector
    k = round(nx * ny * s)
    ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

    # for each color channel
    for j in range(nchan):

        # extract channel
        X = Xorig[:,:,j].squeeze()

        # create images of mask (for visualization)
        Xm = 255 * np.ones(X.shape)
        Xm.T.flat[ri] = X.T.flat[ri]
        masks[i][:,:,j] = Xm

        # take random samples of image, store them in a vector b
        b = X.T.flat[ri].astype(float)

        # perform the L1 minimization in memory
        Xat2 = owlqn(nx*ny, evaluate, None, 5)

        # transform the output back into the spatial domain
        Xat = Xat2.reshape(nx, ny).T # stack columns
        Xa = idct2(Xat)
        Z[i][:,:,j] = Xa.astype('uint8')


plt.imshow(Z, cmap = plt.get_cmap(name = 'gray'))
plt.show()



# X = spimg.zoom(Xorig, 0.1)
# ny, nx = X.shape
# 
# # 50% sample
# k = round(nx * ny * 0.5)
# 
# # random sample of indices
# ri = np.random.choice(nx * ny, k, replace=False)
# 
# b = X.T.flat[ri]
# b = np.expand_dims(b, axis=1)
# 
# # create dct matrix operator using kron (memory errors for large ny*nx)
# A = np.kron(
#     spfft.idct(np.identity(nx), norm='ortho', axis=0),
#     spfft.idct(np.identity(ny), norm='ortho', axis=0)
#     )
# A = A[ri,:]
# 
# # do L1 optimization
# vx = cp.Variable((nx * ny, 1))
# objective = cp.Minimize(cp.norm(vx, 1))
# print(X.shape, A.shape, b.shape)
# constraints = [A*vx == b]
# prob = cp.Problem(objective, constraints)
# result = prob.solve(verbose=True)
# Xat2 = np.array(vx.value).squeeze()
# 
# # reconstruct signal
# Xat = Xat2.reshape(nx, ny).T # stack columns
# Xa = idct2(Xat)
# 
# # confirm solution
# if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
#     print('Warning: values at sample indices don\'t match original.')
# 
# # create images of mask (for visualization)
# mask = np.zeros(X.shape)
# mask.T.flat[ri] = 255
# Xm = 255 * np.ones(X.shape)
# Xm.T.flat[ri] = X.T.flat[ri]
# 
# plt.imshow(Xm, cmap = plt.get_cmap(name = 'gray'))
# plt.show()
