from narnia.core import Driver

import cv2
import numpy as np


@Driver()
class UIDriver():

    def __init__(self):
        super().__init__()
        self.current_frame = np.zeros((1920, 1080))
        self.selected_points = []
        self.refresh()
        cv2.setMouseCallback('window', self.on_click)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.selected_points.append((x, y))
        if event == cv2.EVENT_RBUTTONUP:
            del self.selected_points[-1]

    def refresh(self):
        im = np.copy(self.current_frame)
        [cv2.circle(im, (x, y), 3, (0, 255, 0)) for (x, y) in self.selected_points]
        cv2.imshow('window', im)
        cv2.waitKey(1)

    def show(self, img):
        self.current_frame = img
        self.refresh()

    def close(self):
        cv2.destroyAllWindows()


