
import math
import time
from datetime import datetime

import numpy as np
from cv2 import cv2 as cv
import scipy.fftpack as spfft


def log(s):
    now = datetime.now()
    print("{}: {}".format(now, s))


def dct_matrix(N):
    psi = np.zeros((N, N))
    v = range(N)
    for k in range(N):
        dct_1d = np.cos(np.dot(v, k*math.pi/N))
        if k > 0:
            dct_1d = dct_1d - np.mean(dct_1d)
        psi[:,k] = dct_1d/np.linalg.norm(dct_1d)
    return psi


def dct_1d_matrix(M):
    """
    for k = 1:m
    for n = 1:m
             Phi(k,n) = cos((2*n-1)*(k-1)*pi/(2*m));
    end
    if k==1
             Phi(k,:) = sqrt(1/m).*Phi(k,:);
    else
            Phi(k,:) = sqrt(2./m).*Phi(k,:);
    end
    end
    """
    psi = np.zeros((M, M))
    v = range(1, 2*M+1, 2)
    for i in range(M):
        row = np.cos(np.dot(v, i*np.pi/(2*M)))
        if i == 0:
            row = np.sqrt(1/M) * row
        else:
            row = np.sqrt(2/M) * row
        psi[i,:] = row
    return psi


def dct_2d_matrix(M, N):
    """
    for i = 1:m
        for j = 1:n
            for p = 1:m
                for q = 1:n
                                temp(p,q) = cos((2*p-1)*(i-1)*pi/(2*m))*cos((2*q-1)*(j-1)*pi/(2*n));
                end
            end

            if i==1
                        temp = sqrt(1/m).*temp;
            else
                        temp = sqrt(2/m).*temp;
            end

            if j==1
                        temp = sqrt(1/n).*temp;
            else
                        temp = sqrt(2/n).*temp;
            end

            Phi((j-1)*m+i,:) = temp(:)';
        end
    end
    """
    psi = np.zeros((M*N, M*N))
    tmp = np.zeros((M, N))

    for i in range(M):
        p = range(1, 2*M+1, 2)

        for j in range(N):
            q = range(1, 2*N+1, 2)

            t1 = np.cos(np.dot(p, i*np.pi/(2*M))).reshape(M, 1)
            t2 = np.cos(np.dot(q, j*np.pi/(2*N))).reshape(1, N)
            tmp = t1.dot(t2)

            if i == 0:
                tmp = np.sqrt(1/M) * tmp
            else:
                tmp = np.sqrt(2/M) * tmp

            if j == 0:
                tmp = np.sqrt(1/N) * tmp
            else:
                tmp = np.sqrt(2/N) * tmp

            psi[j*M+i-1,:] = tmp.flat[:]
    return psi


def convert_grayscale_image(path, save=False):
    """
    save a bio image from a existed image with bio_filename
    """
    res = cv.imread(path, 0)
    if save:
        bio_path = "{}/{}{}".format(STATICS_DIR, "gray_", path)
        cv.imwrite(bio_path, res)
    return res


def get_bucket_power_from_file(filepath, N=None):
    """
    load processed output file of power meter
    """
    bs = []
    f = open(filepath, 'r')
    for i, line in enumerate(f.readlines()):
        if N != None and N > 0 and i >= N:
            break
        bs.append(float(line))
    f.close()
    return bs


def matrix_from_image(filepath):
    """
    """
    return cv.imread(filepath, 0)


def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)
