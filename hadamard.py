
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


def generate_speckle(n_speckle=32, disorder=False):
    n = n_speckle**2
    h = hadamard(n)
    h = (h + 1) / 2

    if disorder:
        index = np.random.choice(n, n, replace=False)
        h = h[index]

    h = h * 255

    nds = []
    for row in h:
        nds.append(row.reshape(n_speckle, n_speckle))
    return nds


def save_speckle(nds):
    n = nds[0].shape[0]
    filename = "raw{}.txt".format(n)

    save_raw_list(nds, filename)
    print(nds[0])

    nds = load_raw(filename, (n, n))
    print(nds[0])

    generate_images(filename, (n, n))


def main():
    # generate_speckle()
    save_speckle(generate_speckle(128))


if __name__ == "__main__":
    main()
