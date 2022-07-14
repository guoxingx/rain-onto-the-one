#! /usr/bin/env python
# coding: utf-8

"""
这个文件可单独执行，不包含任何其他自定义文件

更完整的功能请使用gi_tools.py
"""

import os
import time
import json
import socket
import functools
import socketserver
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler, BaseHTTPRequestHandler


Verbose = False

# 积分时间，每次采样会先跳过这段时间，再开始采集数据
INTERGRATION_TIME = 1.5

# 采集时间，每次采集会取这个长度的时间段
SAMPLING_TIME = 0.5

# 允许误差，超过这个值视为不同数据 (0~1)
ALLOW_ERROR = 0.1

# 功率计文件目录
PM_DIR = "../../Documents/StarLab"

# 时间信号目录
SIGNAL_DIR = "."
# 时间信号文件名
SIGNAL_FILE = "signals.txt"

# 输出文件名称
OUTPUT_FILENAME = "output.txt"


def ts2time(ts):
    """
    timestamp to 23:51:16.515000
    """
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")



def sampling(ts, pm_output_file, index):
    return sampling_from_powermeter(ts, pm_output_file)
    log("image at index {} sampling complete".format(index))
    return 0


def filename_required(default=None, note=None):
    if note == None:
        name = input("输入文件名: ")
    else:
        name = input("输入{}: ".format(note))

    if name == "" and default != None:
        name = default

    f = open(name, 'r')
    f.close()
    return name


def extract(pm, ts_list):
    f = open(pm, 'r')
    lines = f.readlines()
    f.close()

    # 获取功率计起始时间
    pm_active_line = lines[31]
    parts = pm_active_line.split("Arrived :")
    tStr = parts[1]
    parts = tStr.split("at")
    dateStr = parts[0].strip()
    timeStr = parts[1].strip()
    t = datetime.strptime("{} {}".format(dateStr, timeStr), "%d/%m/%Y %H:%M:%S")
    pm_active_ts = time.mktime(t.timetuple())
    info("功率计起始时间: {}".format(ts2time(pm_active_ts)))

    # 清楚功率计文件初始信息
    lines = lines[33:]

    # 循环的当前坐标
    current_line, res = 0, []
    for i, ts in enumerate(ts_list):
        lines = lines[current_line:]

        begin_ts = ts + INTERGRATION_TIME
        end_ts = begin_ts + SAMPLING_TIME

        if i % 100 == 0:
            info("extraction, index: {}".format(i))
        else:
            debug("extraction, index: {}".format(i))
        debug("collect data between {} ~ {}".format(ts2time(begin_ts), ts2time(end_ts)))

        update_line, values = 0, []
        for j, line in enumerate(lines):
            s = line.strip()
            parts = s.split(" ")

            d, value = parts[0], parts[-1]
            try:
                d = float(d)
                value = float(value)
            except Exception as e:
                error("failed to convert interval: {}".format(e))
                error("line {}: {} into {}".format(i, line, parts))

            if d + pm_active_ts < begin_ts:
                # only record after INTERGRATION_TIME
                continue

            if d + pm_active_ts > end_ts:
                debug("out of range of time, loop will be finished")
                update_line = j
                break
            values.append(value)

        current_line = update_line
        if len(values) == 0:
            error("falied to load valid value, 0 will be write into output file")
            avg = 0
        else:
            avg = sum([v for v in values]) / len(values)
        res.append(avg)
    return res


def ts_from_recorded_signals(signalfile):
    tss = []
    f = open(signalfile, 'r')
    for i, line in enumerate(f.readlines()):
        ts = float(line)
        if ts > 1000000000000:
            ts = ts / 1000
        tss.append(ts)
    f.close()
    return tss


def log(s):
    now = datetime.now()
    # print("{}: {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), s))
    print("{}: {}".format(now, s))


def error(s):
    print("[ERROR] {}: {}".format(datetime.now(), s))


def info(s):
    print("[INFO] {}: {}".format(datetime.now(), s))


def debug(s):
    if Verbose:
        print("[DEBUG] {}: {}".format(datetime.now(), s))


def mode_required():
    print("鬼像数据提取v1.0")
    print("选择要执行的功能:")
    print("    1 - 快速模式")
    print("    2 - 文件夹模式(默认)")
    print("    3 - 详细模式")
    mode = input("输入 选项: ")

    if mode == "":
        mode = 2
    else:
        mode = int(mode)
    return mode


def run(mode):
    if mode == 1:
        files = os.listdir(PM_DIR)
        files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(PM_DIR, f)))
        pms = [f for f in files if "signals" not in f and "output" not in f and "txt" in f]
        if len(pms) == 0:
            print("没有找到功率计文件！")
            print(PM_DIR, files)
            time.sleep(3)
            return

        files = os.listdir(SIGNAL_DIR)
        files = sorted(files, key=lambda f: os.path.getmtime(f))
        signals = [f for f in files if "signals" in f]
        if len(signals) == 0:
            print("没有找到signals文件！")
            time.sleep(3)
            return

        signalfile, pm_out = os.path.join(SIGNAL_DIR, signals[-1]), os.path.join(PM_DIR, pms[-1])
        outfile = "{}_{}.txt".format(OUTPUT_FILENAME[:-4], datetime.now().strftime("%m%d_%H:%M:%S"))

    elif mode == 2:
        datadir = input("数据文件夹: ")
        datadir = "data/{}".format(datadir.strip("/"))

        files = os.listdir(datadir)
        files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(datadir, f)))
        if "output.txt" in files:
            files.remove("output.txt")

        signals = [f for f in files if "signals" in f and "swp" not in f]
        if len(signals) == 0:
            print("没有找到signals文件！")
            time.sleep(3)
            return

        pms = [f for f in files if "signals" not in f and "output" not in f and "txt" in f]
        if len(pms) == 0:
            print("没有找到功率计文件！")
            time.sleep(3)
            return

        signalfile, pm_out = os.path.join(datadir, signals[-1]), os.path.join(datadir, pms[-1])
        outfile = "{}/{}".format(datadir, OUTPUT_FILENAME)

    elif mode == 3:
        signalfile = filename_required("signals.txt", "散斑信号文件(默认signals.txt)")
        pm_out = filename_required("sample.txt", "功率计文件(默认sample.txt)")
        outfile = "{}_{}.txt".format(OUTPUT_FILENAME[:-4], datetime.now().strftime("%m%d_%H:%M:%S"))

    else:
        print("无效指令")
        time.sleep(3)
        return

    info("时间信号文件: {}".format(signalfile))
    info("功率计文件: {}".format(pm_out))
    info("输出文件: {}".format(outfile))
    print("")

    tss = ts_from_recorded_signals(signalfile)
    out = extract(pm_out, tss)
    f = open(outfile, 'a+')
    for v in out:
        f.write("{}\n".format(v))
    f.close()


if __name__ == "__main__":
    run(mode_required())
