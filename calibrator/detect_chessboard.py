import cv2
import numpy as np
import os
from glob import glob
from collections import defaultdict


def detect_chessboard_points(image_dir, checkerboard_size=(9, 6), square_size=20.0):
    """
    Detects chessboard corners in multi-camera calibration images.

    Parameters:
        image_dir (str): Path to folder containing images.
        checkerboard_size (tuple): Number of internal corners per chessboard row and column.
        square_size (float): Size of each square in mm.

    Returns:
        objpoints (dict): 3D real world points per camera and frame.
        imgpoints (dict): 2D detected points per camera and frame.
        image_shape (tuple): Size of calibration images.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = defaultdict(dict)
    imgpoints = defaultdict(dict)
    image_shape = None

    images = sorted(glob(os.path.join(image_dir, '*.jpg')) + glob(os.path.join(image_dir, '*.png')))

    for fname in images:
        basename = os.path.basename(fname)
        cam_str = basename[:2]  # e.g., '00'
        frame_str = basename[3:9]  # e.g., '000001'
        if int(frame_str) % 2 == 0:
            continue

        img = cv2.imread(fname)
        if image_shape is None:
            image_shape = img.shape[:2][::-1]  # (width, height)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            corners_refined = cv2.cornerSubPix(
                gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)

            objpoints[cam_str][frame_str] = objp.copy()
            imgpoints[cam_str][frame_str] = corners_refined

            # for pt in corners_refined:
            #     cv2.circle(img, tuple(pt.squeeze().astype(np.int32)), 5, (0, 0, 255), 5)  # red: detected

            # cv2.resize(img, (360, 360), interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(os.path.join('debug', basename), img)


    return objpoints, imgpoints, image_shape
