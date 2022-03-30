
import numpy as np
from cv2 import cv2 as cv


# static dir
STATICS_DIR = "statics"

# shadow dir
SHADOW_DIR = "shadow"

# size of shadow
SIZE = (60, 60)

# pixels
PIXELS = (1080, 1080)

# object
Image_Path = "{}/xiongji.png".format(STATICS_DIR)

# sampling times
Sampling_Times = 40000


def get_object_nd(path):
    res = cv.imread(path, 0)
    bio_path = "{}/{}{}".format(STATICS_DIR, "bio_", path)
    cv.imwrite(bio_path, res)
    return res


def get_slm_nd(size):
    # pseudothermal light in Gaussian distribution
    # light = np.random.normal(0, 1, size)
    light = np.random.uniform(0, 255, size)
    return np.zeros(size) + light


def signal_power_from_file(filepath):
    tss = []
    f = open(filepath, 'r')
    for i, line in enumerate(f.readlines()):
        ts = float(line)
        if ts > 1000000000000:
            ts = ts / 1000
        tss.append(ts)
    f.close()
    return tss


def reference_matrix_from_img(filepath):
    res = cv.imread(filepath, 0)

    # if a pixel > 127, set to 255, otherwise set to 0
    ret, res = cv.threshold(res, 127, 255, cv.THRESH_BINARY)
    return res


def reference_matrix_from_img_compress(filepath, a):
    raw = cv.imread(filepath, 0)
    shape = (int(raw.shape[0]/a), int(raw.shape[1]/a))
    res = np.zeros(shape)

    for i, row in enumerate(res):
        for j, col in enumerate(row):
            # res[i, j] = raw[i*a, j*a]
            if raw[i*a, j*a] > 127:
                res[i, j] = 255
            else:
                res[i, j] = 0
    return res


def recovery(signal_powers, images_dir):
    times = len(signal_powers)

    # c_sum = np.zeros((1080, 1080))
    # I_reference_sum = np.zeros((1080, 1080))

    c_sum = np.zeros((40, 40))
    I_reference_sum = np.zeros((40, 40))

    g_sum = 0

    for i, g in enumerate(signal_powers):
        image_name = "{}/p_{}.jpg".format(images_dir, i+1)
        # I_reference = reference_matrix_from_img(image_name)
        I_reference = reference_matrix_from_img_compress(image_name, 27)
        # print("index: {}, refrence: {}".format(i, I_reference))

        I_reference_sum = I_reference_sum + I_reference
        c_sum = c_sum + I_reference * g
        g_sum += g

    g_avg = g_sum / times
    I_reference_avg = I_reference_sum / times
    c_avg = c_sum / times

    gi = c_avg - g_avg * I_reference_avg
    mi = np.min(gi)
    mx = np.max(gi)
    gi = 255 * (gi - mi) / (mx - mi)
    print("gi is {}".format(gi))

    cv.imwrite("{}/result.png".format(STATICS_DIR), gi)


def sampling(obj, times):

    # the correlation result
    # C = <I_signal * I_reference>
    C_sum = np.zeros(obj.shape)

    bucket_sum = 0
    I_SLM_sum = np.zeros(obj.shape)

    for i in range(times):
        I_SLM = get_slm_nd(obj.shape)

        # matrix of signal light
        I_signal = np.multiply(I_SLM, obj)
        # print("\nslm is {}, \nsignal is {}".format(I_SLM, I_signal))
        # I_SLM_sum = I_SLM_sum + I_signal
        I_SLM_sum = I_SLM_sum + I_SLM

        # a number of received power of the bucket detector
        bucket = np.sum(I_signal)
        bucket_sum += bucket

        # C = <I_signal * I_reference>
        C_sum = C_sum + I_SLM * bucket

    bucket_avg = bucket_sum / times
    I_SLM_avg = I_SLM_sum / times
    C_avg = C_sum / times

    gi = C_avg - bucket_avg * I_SLM_avg
    mi = np.min(gi)
    mx = np.max(gi)
    gi = 255 * (gi - mi) / (mx - mi)
    print("obj is {}".format(obj))
    print("gi is {}".format(gi))

    cv.imwrite("{}/result.png".format(STATICS_DIR), gi)


if __name__ == "__main__":
    sp = signal_power_from_file("output.txt")
    recovery(sp, "images40x40")

    # print(reference_matrix_from_img("images/p_1.jpg"))

    # obj = init_object(Image_Path)
    # sampling(obj, Sampling_Times)
