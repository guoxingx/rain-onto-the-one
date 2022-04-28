# coding: utf-8

import time
from datetime import datetime

from utils import log


# 输出结果的文件名
OUTPUT_FILENAME = "output.txt"
SIGNAL_FILE = "signals.txt"

PMActiveTs = None # activate timestamp of power meter

Prev = None


def ts2time(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")


def sampling(ts, pm_output_file, index):
    return sampling_from_powermeter(ts, pm_output_file)
    log("image at index {} sampling complete".format(index))
    return 0


def filename_required(default=None, note=None):
    if note == None:
        name = input("type in 'filename': ")
    else:
        name = input("type in 'filename' of {}: ".format(note))

    if name == "" and default != None:
        name = default

    f = open(name, 'r')
    f.close()
    return name


def empty_samping(ts, pm_output_file):
    """
    ;First Pulse Arrived : 06/03/2022 at 11:24:59
    """
    cs = Sampling()
    cs.begin = 0

    cs.ts_begin = ts + INTERGRATION_TIME
    cs.ts_end = ts + INTERGRATION_TIME + SAMPLING_TIME
    log("collect data between {} ~ {}".format(ts2time(cs.ts_begin), ts2time(cs.ts_end)))

    # time.sleep(INTERGRATION_TIME + SAMPLING_TIME)

    f = open(pm_output_file, 'r')

    begin = False
    for i, line in enumerate(f.readlines()):
        if "First Pulse Arrived" in line:
            # record the activate of pm
            parts = line.split("Arrived :")
            tStr = parts[1]
            parts = tStr.split("at")

            dateStr = parts[0]
            timeStr = parts[1]
            dateStr = dateStr.strip()
            timeStr = timeStr.strip()
            t = datetime.strptime("{} {}".format(dateStr, timeStr), "%d/%m/%Y %H:%M:%S")

            ts = time.mktime(t.timetuple())
            log("find activate time is {}, time: {}, line: {}".format(t, ts2time(ts), i))

            global PMActiveTs
            PMActiveTs = ts
            continue

        if "Timestamp" in line:
            log("find the beginning of data, line: {}".format(i))
            begin = True
            continue

        if not begin:
            continue

        s = line.strip()
        parts = s.split(" ")

        d, value = parts[0], parts[-1]
        try:
            d = float(d)
            value = float(value)
        except Exception as e:
            log("failed to convert interval: {}".format(e))
            log("line {}: {} into {}".format(i, line, parts))

        if d + PMActiveTs < cs.ts_begin:
            # only record after INTERGRATION_TIME
            continue

        if d + PMActiveTs > cs.ts_end:
            log("out of range of time, loop will be finished")
            break

        log("[valid] value: {}, line: {}, time: {}".format(value, i, ts2time(d + PMActiveTs)))
        # record the first and the last line
        if cs.begin == 0:
            cs.begin = i
        if i > cs.end:
            cs.end = i

        cs.values.append(value)

        # only recorded when value is stabled
        if cs.is_stable(value):
            cs.valid_values.append(value)
            if cs.valid_begin == 0:
                cs.valid_begin = i
            cs.valid_end = i

    log("finish loop, ended in line {}".format(i))
    if len(cs.valid_values) == 0:
        log("no valid values found!")
        log("all values: {}".format(cs.values))
        f.close()

    f.close()

    if len(cs.valid_values) == 0:
        log("falied to load valid value, 0 will be write into output file")
        cs.avg = 0
    else:
        cs.avg = sum([v for v in cs.valid_values]) / len(cs.valid_values)
        log("valid line {} to {}, recorded line from {} to {}".format(cs.valid_begin, cs.valid_end, cs.begin, cs.end))
        log("avg: {}, count: {}, valid_values: {}".format(cs.avg, len(cs.valid_values), cs.valid_values))
        print("")

    f = open(OUTPUT_FILENAME, 'a+')
    f.write("\n")
    f.write("[采集结果] 开始时间：{}\n".format(datetime.fromtimestamp(ts)))
    f.close()

    global Prev
    Prev = cs

    return cs.avg


def sampling_from_powermeter(ts, pm_output_file):
    global Prev
    if Prev == None:
        log("first sampling")
        return empty_samping(ts, pm_output_file)

    # record timestamp in the begining
    cs = Sampling()
    cs.ts_begin = time.time()
    cs.end = Prev.end

    cs.ts_begin = ts + INTERGRATION_TIME
    cs.ts_end = ts + INTERGRATION_TIME + SAMPLING_TIME
    log("collect data between time {} ~ {}".format(ts2time(cs.ts_begin), ts2time(cs.ts_end)))

    # time.sleep(INTERGRATION_TIME + SAMPLING_TIME)

    f = open(pm_output_file, 'r')

    log("line befor {} will be skiped".format(Prev.end))
    for i, line in enumerate(f.readlines()):
        if i < Prev.end:
            continue

        s = line.strip()
        parts = s.split(" ")

        d, value = parts[0], parts[-1]
        try:
            d = float(d)
            value = float(value)
        except Exception as e:
            log("failed to convert interval: {}".format(e))
            log("line {}: {} into {}".format(i, line, parts))

        if d + PMActiveTs < cs.ts_begin:
            # only record after INTERGRATION_TIME
            continue

        if d + PMActiveTs > cs.ts_end:
            log("out of range of time, loop will be finished")
            break

        # record the first and the last line
        if cs.begin == 0:
            cs.begin = i
        if i > cs.end:
            cs.end = i

        log("[valid] value: {}, line: {}, time: {}".format(value, i, ts2time(d + PMActiveTs)))
        cs.values.append(value)

        # only recorded when value is stabled
        if cs.is_stable(value):
            cs.valid_values.append(value)
            if cs.valid_begin == 0:
                cs.valid_begin = i
            cs.valid_end = i

    log("finish loop, ended in line {}".format(i))
    if len(cs.valid_values) == 0:
        log("no valid values found!")
        log("all values: {}".format(cs.values))
        f.close()

    f.close()

    if len(cs.valid_values) == 0:
        log("falied to load valid value, 0 will be write into output file")
        cs.avg = 0
    else:
        cs.avg = sum([v for v in cs.valid_values]) / len(cs.valid_values)
        log("valid line {} to {}, recorded line from {} to {}".format(cs.valid_begin, cs.valid_end, cs.begin, cs.end))
        log("avg: {}, count: {}, valid_values: {}".format(cs.avg, len(cs.valid_values), cs.valid_values))
        print("")

    Prev = None
    Prev = cs
    return cs.avg


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


if __name__ == "__main__":
    signalfile= filename_required("signals.txt", "signal file")
    pm_out = filename_required("sample.txt", "power meter")

    f = open(OUTPUT_FILENAME, 'a+')
    tss = ts_from_recorded_signals(signalfile)
    for i, ts in enumerate(tss):
        res = sampling(ts, "sample.txt", i)
        f.write("{}\n".format(res))
    f.close()
