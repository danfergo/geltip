import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

cap = cv2.VideoCapture('/home/danfergo/Videos/Webcam/2020-01-21-172745.webm')
# out = cv2.VideoWriter('out.mpeg', -1, 30.0, (400, 1280))

frame = None
prev = None
past = []
past_diffs = []
kernel = np.ones((3, 3), np.uint8)


def conv2d(x, k):
    dst = cv2.filter2D(x.copy(), -1, k)
    # dst[dst < np.sum(k)] = 0
    # dst / np.sum(k)
    return dst


while (True):
    ret, frame = cap.read()
    if ret is False:
        print('skip.')
        break


    fshape = np.shape(frame)
    frame = cv2.resize(frame, (fshape[1] // 4, fshape[0] // 4,))
    fshape = np.shape(frame)

    fame_color = frame

    frame = np.copy(fame_color)

    frame = cv2.blur(frame, (3, 3))

    past.insert(0, frame)

    if len(past) > 1:
        diff_frame = cv2.absdiff(past[0], past[1])
        # diff_frame[diff_frame <= 15] = 0
        diff_frame[diff_frame <= 5] = 0
        diff_frame = cv2.erode(diff_frame, kernel, iterations=2)
        diff_frame = cv2.dilate(diff_frame, kernel, iterations=2)

        # diff_frame = diff_frame * 0.6 + 0.4 * cv2.dilate(diff_frame, kernel, iterations=1)
        # diff_frame = diff_frame.astype(np.uint8)
        # plt.close()
        cv2.imshow('frame', diff_frame)





        # diff_frame = np.zeros((800, 1280), dtype=np.float32)

        past_diffs.insert(0, diff_frame)

        # acc -= (cv2.absdiff(past[-1], past[-2]) / 255.0)

        acc = np.zeros((fshape[0], fshape[1], 3), dtype=np.float32)
        for diff in past_diffs:
            acc += diff / 255.0

        acc = np.sum(acc, axis=2)
        acc = acc - np.min(acc)
        acc = acc if np.max(acc) == 0 else acc / np.max(acc)

        # acc[acc > 1] = 1
        # acc[acc <= 0.3] = 0
        # acc[acc > 0.3] = 1
        #
        # # acc *= 50
        #
        # x = cv2.resize(acc, (400, 300))
        # x[x > 0.5] = 1
        #
        k = np.ones((3, 3))
        k[1, 1] = 0

        m = acc.copy()
        m[acc > 0] = 1
        dst = cv2.filter2D(m, -1, k)

        acc[dst < 5] = 0

        # x[dst < 10] = 0
        #
        # for i in range(3):
        #     k1 = np.array(
        #         [[0, 0.25, 0.5, 0.25, 0],
        #          [0, 0.50, 1.0, 0.50, 0],
        #          [0, 0.00, 0.0, 0.00, 0],
        #          [0, 0.50, 1.0, 0.50, 0],
        #          [0, 0.25, 0.5, 0.25, 0]])
        #     k2 = np.array(
        #         [[0.25, 0.5, 0, 0.0, 0.00],
        #          [0.50, 1.0, 0, 0.0, 0.00],
        #          [0.00, 0.0, 0, 0.0, 0.00],
        #          [0.00, 0.0, 0, 1.0, 0.50],
        #          [0.00, 0.0, 0, 0.50, 0.25]])
        #
        #     m1 = conv2d(x, k1)
        #     m2 = conv2d(x, k1.T)
        #     m3 = conv2d(x, k2)
        #     m4 = conv2d(x, k2.T)
        #     x += m1 + m2 + m3 + m4
        #
        # # m[m > 1] = 1
        # # shw = m
        # cv2.imshow('better', x)
        #
        acc_disp = (acc * 255).astype(np.uint8)

        # kk =
        # kk2 = np.array([[0, 1, 0],
        #                 [0, 0, 0],
        #                 [0, 1, 0]])
        #
        # dst = cv2.filter2D(acc_disp.copy(), -1, kk)
        # dst[dst < 2] = 0
        # dst = dst / np.sum(kernel)
        # # ret, img = cv2.threshold(dst, 125, 255, cv2.THRESH_BINARY)
        #

        # edges = cv2.Canny(cv2.resize(acc_disp,(300,300)), 125, 255, apertureSize=7)
        # cv2.imshow('edges', edges)

        acc_disp = cv2.dilate(acc_disp, kernel, iterations=2)
        acc_disp = cv2.erode(acc_disp, kernel, iterations=2)
        # acc_disp = cv2.dilate(acc_disp, kernel, iterations=2)
        # acc_disp = cv2.dilate(acc_disp, kernel, iterations=2)

        # cv2.findContours(acc_disp, mode=cv2.cv.CV_RETR_LIST)

        # i
        # umat = cv2.UMat(ij)
        ij = np.array(np.where(acc_disp > 125))
        if len(ij[0]) > 0:
            for i in range(len(ij[0])):
                pass
                # cv2.circle(fame_color, (ij[1, i],ij[0, i]), 1, (0, 0, 255), 3)

        # shape = np.shape(acc_disp)

        # h, w = shape
        # v = np.arange(h)
        # ii = np.tile(v, w).reshape((h, w))
        # jj = np.repeat(v, w).reshape((h, w))
        # ij = np.stack([ii, jj], axis=2)
        #
        # counter = 0
        # areas = [0, ]
        # areas = np.zeros(shape[0] * shape[1])
        # masks = np.zeros(shape)
        # eq = []
        #
        #
        # def f(i, j):
        #     global counter
        #
        #     if acc_disp[i, j] < 0.5:
        #         areas[0] += 1
        #         return
        #
        #     if masks[i, -j]:
        #         counter += 1
        #
        #     masks[i, j] = counter
        #     areas[counter] += 1
        #
        #     print(i, j)
        #
        #
        # [f(ij_[0], ij_[1]) for ij_ in ij]

        # def patch(x, i, j, s):
        #     h, w, _ = np.shape(x)
        #     min_i = np.max(0, i - s)
        #     max_i = w -

        # print(np.shape(ij), ij[5, 9,:])

        # for i in np.nditer(acc_disp):
        #     print(i)

        contours, hierarchy = cv2.findContours(acc_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.contourArea(c) > 30 and cv2.contourArea(c) < 10000]
        print([cv2.contourArea(c) for c in contours])

        cv2.drawContours(fame_color, contours, -1, (0, 255, 0), 3)

        for c in contours:
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(fame_color, ellipse, (0, 0, 255), 2)


        z = np.zeros(fshape[0:2])
        cv2.drawContours(z, contours, -1, 255, -1)
        cv2.imshow('zzz',z)


        # print(contours)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 10000

        # Filter by Circularity
        params.filterByCircularity = False
        # params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        # params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        # params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(acc_disp)
        print('z', keypoints)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        im_with_keypoints = cv2.drawKeypoints(acc_disp, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('xx', im_with_keypoints)

        # mean = tuple(np.average(ij, axis=1).astype(np.uint8).tolist())
        # cv2.circle(fame_color, (mean[1],mean[0]), 15, (255, 0, 0), 3)
        # print(np.shape(mean))
        # print(ij, np.array([[],[]]))
        # print(len(ij[0]))
        # if len(ij[0]) > 0:
        #     umat = np.float32(ij)
        # cv2.fitEllipse(ij)

        # contours, hierarchy = cv2.findContours(acc_disp, cv2.  RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours) > 0:
        #     print(len(contours))
        #     cv2.drawContours(fame_color, contours[0], 3, (0, 255, 0), 3)
        # if len(contours) > 0:
        #     cnt = contours[len(contours) - 1]

        # for contour in contours:
        #     ellipse = cv2.fitEllipse(contour)

        # ij = np.where(acc_disp > 0.5)
        # ellipse = cv2.fitEllipse(ij.T)

        # print(ij)

        # acc_disp = cv2.erode(acc_disp, kernel, iterations=2)
        # acc_disp = cv2.adaptiveThreshold(acc_disp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # acc_disp += cv2.dilate(diff_frame, kernel, iterations=1)

        # Setup SimpleBlobDetector parameters.
        # params = cv2.SimpleBlobDetector_Params()
        #
        # # Change thresholds
        # params.minThreshold = 1
        # params.maxThreshold = 255
        # params.thresholdStep = 10
        # params.minDistBetweenBlobs = 500

        # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 1500

        # Filter by Circularity
        # params.filterByCircularity = True
        # params.minCircularity = 0.1

        # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.87

        # Filter by Inertia
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        # detector = cv2.SimpleBlobDetector(params)
        # detector = cv2.SimpleBlobDetector_create(params)
        # ver = (cv2.__version__).split('.')
        # if int(ver[0]) < 3:
        # else:

        #
        # acc_disp = cv2.cvtColor(acc_disp, cv2.COLOR_GRAY2RGB)
        # keypoints = detector.detect(255 - acc_disp)
        # print(keypoints)
        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
        #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("Keypoints", im_with_keypoints)

        # detected_kp = cv2.drawKeypoints(detected_kp, keypoints, np.array([]), (0, 0, 255),
        #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("Keypoints", detected_kp)

        # print(np.max(acc_disp))
        # acc_disp = cv2.cvtColor(acc_disp, cv2.COLOR_GRAY2RGB)

        h, w = np.shape(acc_disp)

        gray = cv2.cvtColor(fame_color, cv2.COLOR_BGR2GRAY)
        gray -= np.min(gray)
        gray = gray / np.max(gray)

        # print('2.', np.shape(gray), np.max(gray), np.min(gray))
        subsec = gray[100:-100, 100:-100]
        # axes.set_xlim([xmin, xmax])
        # plt.set_xlim([ymin, ymax])

        # plt.hist(subsec.ravel(), 256, [0, 1])
        # plt.show()

        # print('2.', np.shape(subsec), np.max(subsec), np.min(subsec))

        # ij = np.where(subsec > 0.8)
        # print(ij)
        # print(np.shape(ij))

        #
        # clustering = DBSCAN(eps=3, min_samples=2).fit(np.array(ij).T)
        # print(clustering)
        # print(clustering.labels_)
        #
        # acc_disp = cv2.cvtColor(acc_disp, cv2.COLOR_GRAY2BGR)

        gray *= 255
        gray = gray.astype(np.uint8)
        # gray =

        # gray = cv2.medianBlur(gray, 5)
        # cv2.imshow('smooth', gray)
        #
        # edges = cv2.Canny(gray, 300, 400, apertureSize=5)
        # cv2.imshow('edges', edges)
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        # if lines is not None:
        #     for rho, theta in lines[0]:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1000 * (-b))
        #         y1 = int(y0 + 1000 * (a))
        #         x2 = int(x0 - 1000 * (-b))
        #         y2 = int(y0 - 1000 * (a))
        #
        #         cv2.line(fame_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

        #
        # min_radius = int(0.5 * min(h, w))
        #
        # img = cv2.cvtColor(fame_color, cv2.COLOR_BGR2GRAY)
        # img = cv2.medianBlur(img, 5)
        # # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # img = img.astype(np.uint8)
        #
        #
        # cv2.imshow('mask', cimg)
        #
        # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, min_radius,
        #                            param1=50, param2=30, minRadius=min_radius, maxRadius=0)
        # circles = np.uint16(np.around(circles))
        # for i in circles[0, :]:
        #     # draw the outer circle
        #     cv2.circle(fame_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #     # draw the center of the circle
        #     cv2.circle(fame_color, (i[0], i[1]), 2, (0, 0, 255), 3)
        #

        acc_disp = cv2.cvtColor(acc_disp, cv2.COLOR_GRAY2BGR)

        # acc_disp = acc_disp[:, w - h - 200: w, :]
        # fame_color = fame_color[:, w - h - 200: w, :]
        display = np.concatenate([fame_color, acc_disp], axis=1)

        # display = cv2.resize(display, (np.shape(display)[1] // 2, np.shape(display)[0] // 2,))

        # cv2.fitEllipse()

        cv2.imshow('frame', display)

        # Display the resulting frame
        if cv2.waitKey(-1) & 0xFF == ord('q'):
            break

        if len(past) >= 7:
            del past[-1]
            del past_diffs[-1]

# When everything done, release the capture
cap.release()
# out.release()
cv2.destroyAllWindows()
