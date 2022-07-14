
import os
import json
import numpy as np
from cv2 import cv2 as cv

from utils import log
from params import IMAGES_DIR, RAW_DATA_PATH, SIZE, PIXELS


def encode(nd):
    """
    encode a nd array to raw data
    """
    enc = []
    for i, row in enumerate(nd):
        s = 0
        for j, col in enumerate(row):
            if col != 0:
                s += 1 << len(row) - j - 1
        enc.append(s)
    return enc


def decode(array, size=SIZE, bio=True):
    """
    decode a nd array from raw data
    """
    col_length = size[1]
    nd = np.zeros(size)
    for i, rnumber in enumerate(array):
        bin_str = bin(rnumber)[2:]

        bin_str_length = len(bin_str)
        for j_number in range(bin_str_length):
            j = j_number + col_length - bin_str_length
            if int(bin_str[j_number]) != 0:
                if bio:
                    nd[i, j] = 1
                else:
                    nd[i, j] = 255
    return nd


def raw_from_image(filepath, size=SIZE):
    """
    load raw from existed image
    """
    image = cv.imread(filepath, 0)
    size_image = (image.shape[0], image.shape[1])

    if size_image == size:
        return cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    a = int(size_image[0] / size[0])
    b = int(size_image[1] / size[1])

    res = np.zeros(size)
    for i, row in enumerate(res):
        for j, col in enumerate(row):
            if image[i*a, j*b] > 127:
                res[i, j] = 255
    return res


def image_from_ndarray(nd, pixels=PIXELS):
    """
    generate an image nd array in given pixels from raw data
    """
    rate_row = int(pixels[0] / nd.shape[0])
    rate_col = int(pixels[1] / nd.shape[1])

    pixels = (rate_row * nd.shape[0], rate_col * nd.shape[1])

    image = np.zeros(pixels)
    for i in range(nd.shape[0]):
        for j in range(nd.shape[0]):
            if nd[i, j] == 1:
                for ii in range(i*rate_row, (i+1)*rate_row):
                    for jj in range(j*rate_col, (j+1)*rate_col):
                        image[ii, jj] = 255
    return image


def load_raw(filename=None, size=SIZE, bio=True, N=None):
    """
    load raw data from existed file
    """
    if filename == None:
        filename = raw_data_required()
        if filename == None:
            nds = generate_raw()
            save_raw_list(nds)
            return nds


    f = open(filename, 'r')
    nds = []
    for index, line in enumerate(f.readlines()):
        if N != None and N > 0 and index >= N:
            break
        array = json.loads(line)
        dec = decode(array, size=size, bio=bio)
        nds.append(dec)

        if index % 1000 == 0:
            log("load raw index: {}".format(index))
    f.close()
    log("raw data with count {} loaded : {}".format(len(nds), filename))
    return nds


def save_raw_list(nds, filename=None):
    """
    save nd array into raw data
    """
    if filename == None:
        filename = save_raw_data_confirmed()
    log("raw data will be save as {}".format(filename))

    f = open(filename, 'a+')
    for index, nd in enumerate(nds):
        enc = encode(nd)
        f.write(str(enc))
        f.write("\n")

        if index % 1000 == 0:
            log("save raw index: {}".format(index))

    f.close()
    log("raw data with count {} saved into file: {}".format(len(nds), filename))


def generate_raw(size=None):
    """
    generate raw data
    """
    if size == None:
        size = generate_raw_data_confirmed()
    count = size[0] * size[1] * 2
    log("raw data with size {} and count {} will be generated.".format(size, count))

    nds = []
    for index in range(count):
        nd = np.random.binomial(1, 0.5, size)
        nds.append(nd)
    log("generate raw data with count: {}".format(len(nds)))
    return nds


def generate_images(filename=None, size=SIZE):
    """
    generate images with raw data
    """
    nds = load_raw(filename, size=size)

    dir_path = save_images_confirmed()
    if dir_path == None:
        return

    for i, nd in enumerate(nds):
        image = image_from_ndarray(nd)
        cv.imwrite("{}/p_{}.png".format(dir_path, i+1), image)
        if i % 100 == 0:
            log("save images with index: {}".format(i))
    log("{} images saved into dir {}".format(len(nds), dir_path))


def mode_required():
    mode_str = input(
        """
        chose the function to run:
            1. Generate images. (default)
            2. Generate raw data.
            3. Save raw from images

        """
    )

    mode = 1
    if len(mode_str) == 0:
        return mode
    try:
        mode = int(mode_str)
        return mode
    except Exception as e:
        log("invalid input")
        exit(0)


def save_raw_data_confirmed():
    inpt = input(
        """
        save raw data? ["filename" / n]
            type in the filename of raw data to save, default as "raw.txt",
            otherwise type "n" for NOT SAVE

        """
    )

    filename = RAW_DATA_PATH
    if len(inpt) == 0:
        return filename

    if inpt.lower() == "n":
        return None
    return inpt.strip()


def raw_data_required():
    inpt = input(
        """
        load raw data? ["filename" / n]
            type in the filename of existed raw data, default as "raw.txt",
            otherwise type "n" for NEW ONE

        """
    )

    filename = RAW_DATA_PATH
    if len(inpt) == 0:
        return filename

    if inpt.lower() == "n":
        return None
    return inpt.strip()


def generate_raw_data_confirmed():
    inpt = input(
        """
        type in the size for shadow (default: 60):

        """
    )

    if len(inpt) == 0:
        return SIZE

    try:
        size = int(inpt)
        return (size, size)
    except Exception as e:
        log("invalid input")
        exit(0)


def save_images_confirmed():
    inpt = input(
        """
        save images? ["dir name" / n]
            type in the dir name for images to save, default as "images",
                (if "dir name" already existed, every file under the dir will be deleted.)
            otherwise type "n" for NOT SAVE

        """
    )

    path = IMAGES_DIR
    if len(inpt) != 0:
        path = inpt

    if inpt.lower() == "n":
        return None

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for p in os.listdir(path):
            os.remove("{}/{}".format(path, p))
    return path


def dir_required():
    inpt = input(
        """
        images dir:

        """
    )
    return inpt


if __name__ == "__main__":
    mode = mode_required()

    if mode == 1:
        generate_images()

    if mode == 2:
        generate_raw()

    if mode == 3:
        d = dir_required()
        rs = []
        for i in range(3200):
            raw = raw_from_image("{}/p_{}.jpg".format(d, i+1), (40,40))
            if i % 100 == 0:
                print("load raw: {}".format(i))
            rs.append(raw)
        save_raw_list(rs, "raw40.txt")
