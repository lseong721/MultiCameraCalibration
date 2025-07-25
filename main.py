from calibrator.detect_chessboard import detect_chessboard_points
from calibrator.calibrate_camera import calibrate_individual_cameras, filter_points_with_ransac_pnp
from calibrator.extrinsics import estimate_extrinsics_via_best_path
from calibrator.optimize import optimize_bundle
from calibrator.visualize import draw_cameras_and_points
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Camera Calibration and Bundle Adjustment")
    parser.add_argument("--image_dir", type=str, default="D:/Workspace/Dataset/Multi-camera calibration/selected_images_2400/resize_720",
                        help="Directory containing calibration images (e.g., 00_000001.png).")
    parser.add_argument("--checkerboard", type=int, nargs=2, default=(9, 6), 
                        help="Number of inner corners (cols rows) in the checkerboard pattern. Default: 9 6")
    parser.add_argument("--square_size", type=float, default=80.0,
                        help="Size of each square on the checkerboard in mm. Default: 80.0")
    parser.add_argument("--optimize_intrinsic", action="store_true", 
                        help="If set, bundle adjustment will also optimize intrinsic parameters.")
    parser.add_argument("--ref_cam", type=str, default="00", 
                        help="Reference camera ID. Default: 00")
    parser.add_argument("--ransac_threshold", type=float, default=4.0,
                        help="Threshold for RANSAC-based outlier filtering (in pixels). Default: 4.0")
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = args.image_dir
    checkerboard_size = tuple(args.checkerboard)
    square_size = args.square_size
    optimize_intrinsic = args.optimize_intrinsic
    ref_cam = args.ref_cam

    print(f"[1] Detecting chessboard corners in {image_dir}...")
    objpoints, imgpoints, image_shape = detect_chessboard_points(image_dir, checkerboard_size, square_size)

    print("[2] Calibrating individual cameras...")
    camera_params = calibrate_individual_cameras(objpoints, imgpoints, image_shape)

    print("[3] Removing outlier corner points using RANSAC...")
    camera_params = filter_points_with_ransac_pnp(camera_params, objpoints, imgpoints, threshold=args.ransac_threshold)

    print(f"[4] Estimating relative extrinsics with reference camera '{ref_cam}'...")
    extrinsics = estimate_extrinsics_via_best_path(camera_params, objpoints, imgpoints, ref_cam=ref_cam)

    print("[5] Optimizing with bundle adjustment...")
    result, extrinsics = optimize_bundle(camera_params, extrinsics, objpoints, imgpoints, optimize_intrinsic=optimize_intrinsic)

    print("[6] Visualizing cameras and calibration board...")
    draw_cameras_and_points(camera_params, extrinsics, objpoints, checkerboard_size, ref_cam=ref_cam)


if __name__ == "__main__":
    main()
