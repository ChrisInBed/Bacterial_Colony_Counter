# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, morphology, transform, filters, feature, measure, color
import cv2


def cut(img, sigma=0.1, plate_width=40):
    # img = cv2.imread('5B3.Tif', cv2.IMREAD_GRAYSCALE) / 255.0
    im = img.copy()
    im = transform.resize(im, (550, 672))
    contour = feature.canny(im, sigma=sigma)
    blotmap = morphology.dilation(contour, selem=morphology.disk(3))
    radius = np.arange(210, 275, step=3)
    hough = transform.hough_circle(blotmap, radius=radius)
    plate_index = np.unravel_index(hough.argmax(), hough.shape)
    c = draw.circle(*plate_index[1:], radius[plate_index[0]] - plate_width)
    mask = np.zeros_like(im)
    mask[c[0], c[1]] = 1.
    blotmap = blotmap * mask
    return blotmap


def recognize(blotmap):
    label = measure.label(blotmap, connectivity=1)
    blur_map = filters.gaussian(blotmap, 2)
    peaks = feature.peak_local_max(blur_map, min_distance=2, labels=label, num_peaks_per_label=10)
    # peaks = feature.peak_local_max(blur_map, min_distance=2)
    return peaks.shape[0], peaks


def graph(img, peaks, show=True, fname=None):
    im = img.copy()
    im = transform.resize(im, (550, 672))
    im = color.gray2rgb(im)

    fig, ax = plt.subplots(dpi=150)
    ax.imshow(im)
    ax.scatter(peaks[:, 1], peaks[:, 0], c='', edgecolors='r', s=20, linewidths=0.3)
    ax.text(75, 520, 'count: {}'.format(peaks.shape[0]), color='white')
    plt.axis('off')
    if show:
        plt.show()
    # for y, x in peaks:
    #     c = draw.circle_perimeter(y, x, 4)
    #     im[c[0], c[1]] = [1., 0., 0.]
    # text = 'count: {}'.format(peaks.shape[0])

    if fname:
        fig.savefig(fname)


if __name__ == '__main__':
    img = cv2.imread('0B3.Tif', cv2.IMREAD_GRAYSCALE) / 255.0
    plate = cut(img)
    num, dots = recognize(plate)
    graph(img, dots, fname='test.png')
    # graph(plate, dots)
    print(num)
