import numpy as np
import cv2

hscale = 10

def onChange(x):
    global hscale
    hscale = x

def HSVmap():
    hsvmap = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsvmap.shape[:2])
    hsvmap[:, :, 0] = h
    hsvmap[:, :, 1] = s
    hsvmap[:, :, 2] = 255
    hsvmap = cv2.cvtColor(hsvmap, cv2.COLOR_HSV2BGR)
    return hsvmap

def hist2D():
    cap = cv2.VideoCapture('genji.mp4')

    while True:
        ret, img_color = cap.read()
        if ret == False:
            continue
        cv2.imshow('video', img_color)

        hsvmap = HSVmap()

        cv2.namedWindow('hist2D', 0)
        cv2.createTrackbar('scale', 'hist2D', hscale, 32, onChange)

        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist = np.clip(hist * 0.005 * hscale, 0, 1)
        hist = hsvmap * hist[:, :, np.newaxis] / 255.0

        cv2.imshow('hist2D', hist)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()

hist2D()
cv2.waitKey(0)
cv2.destroyAllWindows()















