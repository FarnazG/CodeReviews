import numpy as np
import cv2
import json

def wait_for_esc():
    while 1:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # if k == ord('m'):
        #    mode = not mode
        # elif k == 27:
        #    break


def intersect_lidar_beam(x, y, deg):
    global map_rgb_org

    c = np.cos(np.deg2rad(deg))
    s = np.sin(np.deg2rad(deg))

    for r in range(1, 1000, 9):
        xr = x + int(r * c)
        yr = y + int(r * s)
        if map_rgb_org[yr, xr, 0] == 0:
            return xr, yr, r
    return x, y, 0

