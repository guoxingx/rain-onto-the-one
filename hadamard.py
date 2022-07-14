
"""

np.random.choice(target, n, replace=True)
: 从a中选n个数, replace=True表示可以重复
    a若是int，则从0~a中选整数
     若是数组，则从a数组中选，此时不限定整数


scipy.linalg.hadamard(n)
: 直接生成哈达玛矩阵，n必须是2的阶乘

"""

import numpy as np
from scipy.linalg import hadamard
from generator import save_raw_list, load_raw, generate_images


def generate_speckle(n_speckle=32, index=None):
    n = n_speckle**2
    h = hadamard(n)
    h = (h + 1) / 2

    if index:
        h = h[index]

    nds = []
    for row in h:
        nds.append(row.reshape(n_speckle, n_speckle))
    return nds


def save_speckle(n_speckle, index=None):
    """
    """
    filename = "raw_h{}.txt".format(n_speckle)

    save_raw_list(None, filename, hadamard=index if index is not None else n_speckle)

    generate_images(filename, (n_speckle, n_speckle))


def main():
    n = 64
    # save_speckle(n, index=np.random.choice(n, n, replace=False).tolist())
    save_speckle(n)


if __name__ == "__main__":
    main()
