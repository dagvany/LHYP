from matplotlib import pyplot as plt
import numpy as np


def draw_contourmtcs2image(image, contours, rgbs):
    img_mtx = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(float)
    p1, p99 = np.percentile(img_mtx, (1, 99))
    img_mtx[img_mtx < p1] = p1
    img_mtx[img_mtx > p99] = p99
    img_mtx = (img_mtx - p1) / (p99 - p1)
    for contour, rgb in zip(contours, rgbs):
        x = contour[:, 0].astype(int)
        y = contour[:, 1].astype(int)
        c = np.array(rgb, dtype=float)
        img_mtx[y, x, :] = c
        img_mtx[y+1, x, :] = c
        img_mtx[y, x+1, :] = c
        img_mtx[y+1, x+1, :] = c
    plt.imshow(img_mtx)
    plt.show()
