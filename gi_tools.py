#! /usr/bin/env python
# coding: utf-8

"""
"""

from utils import get_bucket_power_from_file, normalize
from generator import load_raw, save_raw_list

from conventional import conventional
from differential import differential


def multi():
    filedir = input("输入 目标文件夹: ")
    filedir = filedir[:-1] if filedir.endswith("/") else filedir
    filedir = "data/{}".format(filedir)

    b_list = get_bucket_power_from_file("{}/output.txt".format(filedir))
    if len(b_list) <= 1024:
        size = (32, 32)
        r_list = load_raw("raw_h32.txt", size=size, bio=False)
    elif len(b_list) <= 3200:
        size = (40, 40)
        r_list = load_raw("raw_r40.txt", size=size, bio=False)
    elif len(b_list) <= 4096:
        size = (64, 64)
        r_list = load_raw("raw_h64.txt", size=size, bio=False)
    elif len(b_list) <= 7200:
        size = (60, 60)
        r_list = load_raw("raw_r60.txt", size=size, bio=False)
    else:
        size = (128, 128)
        r_list = load_raw("raw_h128.txt", size=size, bio=False)

    bucket_list, reference_list = b_list[:], r_list[:]
    bucket_list = normalize(bucket_list)

    gi = conventional(bucket_list, reference_list, size)
    cv.imwrite("{}/convential.png".format(filedir), gi)

    dgi = differential(bucket_list, reference_list, size)
    cv.imwrite("{}/differential.png".format(filedir), dgi)

    plt.title("功率计响应点 亮光环境")

    plt.subplot(2,2,1)
    plt.title("GI")
    plt.imshow(gi)

    plt.subplot(2,2,2)
    plt.title("DGI")
    plt.imshow(dgi)

    plt.subplot(2,2,3)
    plt.title("功率计采点")
    plt.plot(bucket_list)

    plt.show()


def mode_required():
    print("鬼像数据提取v2.0")
    print("选择要执行的功能:")
    print("    1 - 快速模式")
    print("    2 - 文件夹模式(默认)")
    print("    3 - 详细模式")
    print("    4 - 图像重构")
    mode = input("输入 选项: ")

    if mode == "":
        mode = 2
    else:
        mode = int(mode)
    return mode


def type_required():
    print("选择重构算法:")
    print("    1 - 多种(默认)")
    print("    2 - 关联成像")
    print("    3 - 差分关联")
    print("    4 - 压缩感知")
    mode = input("输入 选项: ")

    if mode == "":
        mode = 1
    else:
        mode = int(mode)
    return mode


if __name__ == "__main__":
    mode = mode_required()

    if mode < 4:
        import ymc
        ymc.run(mode)
        exit(0)

    if mode == 4:
        import matplotlib.pyplot as plt
        from cv2 import cv2 as cv

        plt.rcParams["font.sans-serif"] = "Arial Unicode MS"

        multi()
        # typ = type_required()
        # if typ == 1:
        #     multi()
        #     exit(0)

    else:
        print("无效指令")
        exit()
