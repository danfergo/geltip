import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

BUFFER_SIZE = 50


def resize(i, ratio):
    h, w, d = np.shape(i)
    return cv2.resize(i, (int(w * ratio), int(h * ratio)))


def conv2d(x, k):
    dst = cv2.filter2D(x, -1, np.array(k))
    return dst


def conv(x, ks):
    return np.stack([conv2d(x, ki) for ki in ks], axis=2)


def normalize_img(i):
    i = i - np.min(i)
    i = i if np.max(i) == 0 else i / np.max(i)
    return i


def display(*frames):
    n_rows = math.floor(math.sqrt(len(frames)))
    n_cols = len(frames) // n_rows

    def ff(f):
        if f.dtype != np.uint8:
            f = normalize_img(f) * 255
            f = f.astype(np.uint8)

        if len(f.shape) == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)

        return f

    rows = [np.concatenate([ff(frames[r * n_cols + c]) for c in range(n_cols)], axis=1) for r in range(n_rows)]

    full_frame = np.concatenate(rows, axis=0)

    h, w, d = np.shape(full_frame)
    ww = 2000
    hh = int((h * ww) / w)
    full_frame = cv2.resize(full_frame, (ww, hh))

    cv2.imshow('full frames', full_frame)


def diff(frame1, frame2):
    kernel = np.ones((5, 5), np.float32) / 25
    dst1 = cv2.filter2D(frame1, -1, kernel)

    kernel = np.ones((5, 5), np.float32) / 25
    dst2 = cv2.filter2D(frame2, -1, kernel)

    diff = cv2.absdiff(dst1, dst2)

    s = np.sum(diff, axis=2)
    # print(np.max(s), np.min(s))
    # plt.hist(s.ravel(), 256, [0, 50])
    #
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()

    diff[s < 15] = 0

    diff = cv2.dilate(diff, kernel, iterations=2)
    diff = cv2.erode(diff, kernel, iterations=2)

    return diff


def main():
    cap = cv2.VideoCapture('../data/video2.webm')

    past = []
    past_diffs = []
    sensor_mask = None

    while True:
        # pulling video frame.
        ret, frame = cap.read()
        frame = resize(frame, 0.25)

        if sensor_mask is None:
            sensor_mask = np.zeros(frame.shape[0:2])
            center = (188, 110)
            radius = 125
            m = cv2.circle(sensor_mask, center, radius, 1, -1)
            sensor_mask = np.stack([m, m, m], axis=2)

        frame = sensor_mask * frame

        if ret is False:
            print('skip.')
            break

        past.insert(0, frame)

        if len(past) > 0 and len(past_diffs) > 0:
            detect(frame, past, past_diffs)

        if len(past) > 1:
            past_diffs.insert(0, diff(past[0], past[1]))

        # Display the resulting frame
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

        if len(past) >= BUFFER_SIZE:
            del past[-1]
            del past_diffs[-1]


def density(contour, img):
    z = np.zeros(np.shape(img))
    mask = cv2.drawContours(z, [contour], -1, 1, -1)
    cv2.imshow('mask', mask)
    d = np.sum(mask * img) / np.sum(mask)
    print(d)
    return d
    return


sensor_mask = None


def detect(frame, past, past_diffs):
    fshape = np.shape(frame)

    acc = np.zeros((fshape[0], fshape[1], 3), dtype=np.float32)
    for diff in past_diffs[0:5]:
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(diff, -1, kernel)

        acc += dst / 255.0

    acc = np.sum(acc, axis=2)
    acc = acc - np.min(acc)
    acc = acc if np.max(acc) == 0 else acc / np.max(acc)

    acc = cv2.dilate(acc, kernel, iterations=2)
    acc = cv2.erode(acc, kernel, iterations=2)

    accd = (acc * 255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(accd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if
                cv2.contourArea(c) > 30 and cv2.contourArea(c) < 10000 and density(c, acc) > 0.2]

    canvas = frame.copy()
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 3)

    for c in contours:
        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(canvas, ellipse, (0, 0, 255), 2)

    display(frame, canvas, accd)


if __name__ == '__main__':
    main()
