# coding: utf-8

import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = "Arial Unicode MS"


def read_pm(pmfile):
    times, values = [], []
    f = open(pmfile, "r")
    for i, line in enumerate(f):
        if i < 34:
            continue

        s = line.strip()
        parts = s.split(" ")

        d, value = float(parts[0]), float(parts[-1])
        times.append(d)
        values.append(value)
    f.close()
    return times, values


def run(pmfile):
    _, values = read_pm(pmfile)

    counts = len(values)

    plt.title("功率计响应点 - 亮光环境")

    # plt.subplot(121)
    plt.xlabel("数据点")
    plt.ylabel("读数")
    plt.plot(values)
    prev = 0
    marks = []
    for i, v in enumerate(values):
        if v != prev:
            prev = v
            marks.append(i)
    plt.plot(values, markevery=marks, ls="", marker="*")

    # # plt.subplot(122)
    # plt.xlabel("数据点")
    # plt.ylabel("读数")
    # plt.plot(values2)
    # prev = 0
    # marks = []
    # for i, v in enumerate(values2):
    #     if v != prev:
    #         prev = v
    #         marks.append(i)
    # plt.plot(values2, markevery=marks, ls="", marker="*")

    plt.show()


if __name__ == "__main__":
    # run("data/0601_10_B_1/output.txt")
    run("data/outdated/0624_19_B_1/963295_06.txt")
