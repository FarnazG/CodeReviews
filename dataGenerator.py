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


def synthetic_lidar_generator(x, y, theta):
    global map_rgb

    lidar_point_cloud = []
    for deg in range(theta, 360+theta, 5):
        xd, yd, d = intersect_lidar_beam(x, y, deg)
        lidar_point_cloud.append([int(d * np.cos(np.deg2rad(deg-theta))), int(d * np.sin(np.deg2rad(deg-theta)))])
        cv2.line(map_rgb, (x, y), (xd, yd), (0, 0, 255), 1)

    return lidar_point_cloud


# mouse callback function
def draw_circle(event, x, y, flags, params):
    global drawing, position, mapname, map_rgb, map_rgb_org

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        veh_dict = {'veh_pos': [x, y, 0.0], 'point_cloud': 0}
        position.append(veh_dict.copy())
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            last = position[-1]['veh_pos']
            if np.linalg.norm([x - last[0], y - last[1]]) < 10:
                return

            # print("norm: ", np.linalg.norm([x - last[0], y - last[1]]))
            # print("(xt-1,yt-1): (", last[0], ", ", last[1], ") - (x,y): (", x, ",", y, ")")

            theta = np.arctan2(y - last[1], x - last[0])
            map_rgb = np.copy(map_rgb_org)
            lidar_point_cloud = synthetic_lidar_generator(x, y, int(np.rad2deg(theta)))
            start_point = (x, y)
            end_point = (x + int(30.0 * np.cos(theta)), y + int(30.0 * np.sin(theta)))
            cv2.arrowedLine(map_rgb, start_point, end_point, (0, 0, 0), 3)
            cv2.imshow(mapname, map_rgb)
            veh_dict = {'veh_pos': [x, y, int(np.rad2deg(theta))], 'point_cloud': lidar_point_cloud}
            position.append(veh_dict.copy())
            cv2.waitKey(1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def draw_route(map_name, route_name):
    layout = cv2.imread(map_name)

    with open(route_name, 'r') as F:
        L2 = json.loads(F.read())
    F.close()

    for pos in L2:
        x = pos['veh_pos'][0]
        y = pos['veh_pos'][1]
        theta = np.deg2rad(pos['veh_pos'][2])
        cv2.circle(layout, (x, y), 1, (0, 255, 0), -1)
        end_point = (x + int(30.0 * np.cos(theta)), y + int(30.0 * np.sin(theta)))
        cv2.arrowedLine(layout, (x, y), end_point, (0, 0, 0), 1)

    cv2.imshow(map_name, layout)
    return layout


def replay_route(map_name, route_name):
    layout_org = cv2.imread(map_name)

    with open(route_name, 'r') as F:
        L2 = json.loads(F.read())
    F.close()

    for pos in L2:
        layout = np.copy(layout_org)
        x = pos['veh_pos'][0]
        y = pos['veh_pos'][1]
        theta = np.deg2rad(pos['veh_pos'][2])
        cv2.circle(layout, (x, y), 1, (0, 255, 0), -1)
        end_point = (x + int(30.0 * np.cos(theta)), y + int(30.0 * np.sin(theta)))
        cv2.arrowedLine(layout, (x, y), end_point, (0, 0, 0), 2)

        plc = pos['point_cloud']
        for pnt in plc:
            deg = np.arctan2(pnt[1], pnt[0]) + theta
            r = np.linalg.norm([pnt[0], pnt[1]])
            cv2.line(layout, (x, y), (int(x + r * np.cos(deg)), int(y + r * np.sin(deg))), (0, 0, 255), 1)

        cv2.imshow(map_name, layout)
        cv2.waitKey(33)

