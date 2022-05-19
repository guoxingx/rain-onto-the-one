
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    threshold = 10

    img = cv2.imread("statics/gray_xiongji.jpeg", 0).astype(float)
    img = cv2.resize(img, (378, 378))

    img_dct = cv2.dct(img)
    for i, v in enumerate(img_dct.flat):
        if np.abs(v) < threshold:
            img_dct.flat[i] = 0

    # tmp = cv2.dct(img)
    # img_dct = np.zeros((378, 378))
    # for i, v in enumerate(img_dct.flat):
    #     if i > 378 * 378 * 1 / 4:
    #         break
    #     img_dct.flat[i] = tmp.flat[i]

    # print(img_dct)
    # img_dct = cv2.threshold(img_dct, 0.01, 0, cv2.THRESH_TOZERO)[1]
    # print(img_dct)

    img_idct = cv2.idct(img_dct)
    cv2.imwrite("statics/dct_{}.png".format(threshold), img_idct)

    plt.figure(6, figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(img, 'gray')
    plt.title('original image')

    plt.subplot(232)
    plt.imshow(img_dct, 'gray')
    plt.title('dct')

    plt.subplot(233)
    plt.imshow(img_idct, 'gray')
    plt.title('idct')

    plt.show()
