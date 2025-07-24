import os
from calibrator.detect_chessboard import detect_chessboard_points
from calibrator.calibrate_camera import calibrate_individual_cameras, filter_points_with_ransac_pnp
from calibrator.extrinsics import estimate_extrinsics_via_best_path
from calibrator.optimize import optimize_bundle
from calibrator.visualize import draw_cameras_and_points
import numpy as np


def main():
    image_dir = "D:/Workspace/Dataset/Multi-camera calibration/selected_images_2400/resize_720"  # imge format: 00_000001.png 
    checkerboard_size = (9, 6)  # (9 inner corners in x, 6 inner corners in y)
    square_size = 80.0  # mm

    print("[1] Detecting chessboard corners...")
    objpoints, imgpoints, image_shape = detect_chessboard_points(
        image_dir, checkerboard_size, square_size)

    print("[2] Calibrating individual cameras...")
    camera_params = calibrate_individual_cameras(objpoints, imgpoints, image_shape)

    print("[3] Removing outlier corner points...")
    camera_params = filter_points_with_ransac_pnp(camera_params, objpoints, imgpoints, threshold=4.0)

    print("[4] Estimating relative extrinsics...")
    extrinsics = estimate_extrinsics_via_best_path(camera_params, objpoints, imgpoints, ref_cam='00')

    print("[5] Optimizing with bundle adjustment...")
    result, extrinsics = optimize_bundle(camera_params, extrinsics, objpoints, imgpoints, optimize_intrinsic=False)

    print("[6] Visualizing cameras and calibration board...")
    draw_cameras_and_points(camera_params, extrinsics, objpoints, checkerboard_size, ref_cam='00')



if __name__ == "__main__":
    main()
