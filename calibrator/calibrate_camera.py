import cv2
import numpy as np

def calibrate_individual_cameras(objpoints_dict, imgpoints_dict, image_shape):
    """
    Calibrate each camera independently using its detected chessboard points.

    Parameters:
        objpoints_dict (dict): 3D object points per camera and frame.
        imgpoints_dict (dict): 2D image points per camera and frame.
        image_shape (tuple): Image shape (width, height).

    Returns:
        camera_params (dict): Intrinsic matrix, distortion, and pose per camera.
    """
    camera_params = {}

    for cam_id in objpoints_dict.keys():
        objpoints = []
        imgpoints = []

        for frame_id in objpoints_dict[cam_id].keys():
            objpoints.append(objpoints_dict[cam_id][frame_id])
            imgpoints.append(imgpoints_dict[cam_id][frame_id])

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None)

        camera_params[cam_id] = {
            'camera_matrix': K,
            'dist_coeffs': dist,
            'rvecs': {f: r for f, r in zip(objpoints_dict[cam_id].keys(), rvecs)},
            'tvecs': {f: t for f, t in zip(objpoints_dict[cam_id].keys(), tvecs)}
        }

    return camera_params


def filter_points_with_ransac_pnp(camera_params, objpoints_dict, imgpoints_dict, threshold=4.0):
    """
    Use cv2.solvePnPRansac to detect and remove 2D-3D outlier matches in each frame.
    Updates objpoints_dict and imgpoints_dict in-place.
    Also logs per-camera average reprojection error before and after filtering.
    """
    for cam_id in objpoints_dict:
        K = camera_params[cam_id]['camera_matrix']
        dist = camera_params[cam_id]['dist_coeffs']
        init_errors = []
        final_errors = []
        for frame_id in list(objpoints_dict[cam_id].keys()):
            objp = objpoints_dict[cam_id][frame_id]
            imgp = imgpoints_dict[cam_id][frame_id]

            if objp.shape[0] < 6:
                continue

            # Compute initial error
            proj_init, _ = cv2.projectPoints(objp, camera_params[cam_id]['rvecs'][frame_id],
                                             camera_params[cam_id]['tvecs'][frame_id], K, dist)
            err_init = np.linalg.norm(proj_init.squeeze() - imgp.squeeze(), axis=1)
            init_errors.extend(err_init.tolist())

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objp, imgp, K, dist,
                reprojectionError=threshold,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE)

            if not success or inliers is None or len(inliers) < 10:
                print(f"[RANSAC] Removing frame {cam_id}-{frame_id} (insufficient inliers)")
                del objpoints_dict[cam_id][frame_id]
                del imgpoints_dict[cam_id][frame_id]
                continue

            inliers = inliers.flatten()
            objp_filtered = objp[inliers]
            imgp_filtered = imgp[inliers]

            # Recompute error after filtering
            proj_final, _ = cv2.projectPoints(objp_filtered, rvec, tvec, K, dist)
            err_final = np.linalg.norm(proj_final.squeeze() - imgp_filtered.squeeze(), axis=1)
            final_errors.extend(err_final.tolist())

            objpoints_dict[cam_id][frame_id] = objp_filtered
            imgpoints_dict[cam_id][frame_id] = imgp_filtered

            # update pose with inlier-based refinement
            camera_params[cam_id]['rvecs'][frame_id] = rvec
            camera_params[cam_id]['tvecs'][frame_id] = tvec

        if init_errors and final_errors:
            print(f"[RANSAC] Camera {cam_id}: Mean error before = {np.mean(init_errors):.2f} px, after = {np.mean(final_errors):.2f} px")
    return camera_params
