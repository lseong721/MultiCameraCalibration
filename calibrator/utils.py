import numpy as np
import cv2
import os

def get_board_points_in_world(objpoints_dict, camera_params, ref_cam='00'):
    """
    Transform detected board points from camera coordinate system
    to world coordinates using a reference camera's extrinsic parameters.

    Parameters:
        objpoints_dict (dict): 3D object points [cam][frame] = (N, 3)
        camera_params (dict): Calibration results including rvecs/tvecs
        ref_cam (str): Camera ID to use as world reference

    Returns:
        board_world_points (np.ndarray): Transformed board points in world frame (N, 3)
    """
    # first camera's object points
    frame_id = list(objpoints_dict[ref_cam].keys())[0]
    board_pts = objpoints_dict[ref_cam][frame_id]  # (N, 3)

    rvec = camera_params[ref_cam]['rvecs'][frame_id]
    tvec = camera_params[ref_cam]['tvecs'][frame_id]
    R, _ = cv2.Rodrigues(rvec)

    board_world = (R @ board_pts.T + tvec).T  # (N, 3)
    return board_world


# def project_points_numpy(X, rvec, tvec, K, dist):
#     assert X.shape[1] == 3 and len(X.shape) == 2
#     """
#     X: (N, 3) 3D points in world
#     rvec: (3, 1) rotation vector
#     tvec: (3, 1) translation vector
#     K: (3, 3) camera matrix
#     dist: (5,) distortion coefficients [k1, k2, p1, p2, k3]

#     Returns:
#         projected points: (N, 2)
#     """
#     dist = np.pad(dist.flatten(), (0, max(0, 5 - len(dist))), 'constant')[:5]

#     R, _ = cv2.Rodrigues(rvec)
#     X_cam = (R @ X.T).T + tvec.reshape(1, 3)

#     x = X_cam[:, 0] / (X_cam[:, 2] + 1e-10)  # avoid division by zero   
#     y = X_cam[:, 1] / (X_cam[:, 2] + 1e-10)  # avoid division by zero   
#     r2 = x**2 + y**2

#     k1, k2, p1, p2, k3 = dist
#     radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
#     x_dist = x * radial + 2*p1*x*y + p2*(r2 + 2*x**2)
#     y_dist = y * radial + p1*(r2 + 2*y**2) + 2*p2*x*y

#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]
#     u = fx * x_dist + cx
#     v = fy * y_dist + cy

#     return np.stack([u, v], axis=1)


# def evaluate_reprojection_error(camera_params, objpoints_dict, imgpoints_dict):
#     """
#     Return dict of per-frame mean reprojection error.
#     """
#     error_dict = {}
#     for cam_id in camera_params:
#         cam_errors = []
#         K = camera_params[cam_id]['camera_matrix']
#         dist = camera_params[cam_id]['dist_coeffs']
#         for frame_id in objpoints_dict[cam_id]:
#             objp = objpoints_dict[cam_id][frame_id]
#             imgp = imgpoints_dict[cam_id][frame_id]
#             rvec = camera_params[cam_id]['rvecs'][frame_id]
#             tvec = camera_params[cam_id]['tvecs'][frame_id]
#             proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
#             proj = proj.squeeze()
#             error = np.linalg.norm(proj - imgp.squeeze(), axis=1).mean()
#             cam_errors.append(error)
#         error_dict[cam_id] = cam_errors
#     return error_dict


def draw_projected_corners(camera_params, objpoints_dict, imgpoints_dict, image_dir, out_dir="debug"):
    os.makedirs(out_dir, exist_ok=True)
    for cam_id in objpoints_dict:
        for frame_id in objpoints_dict[cam_id]:
            img_name = f"{cam_id}_{frame_id}.png"
            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            K = camera_params[cam_id]['camera_matrix']
            dist = camera_params[cam_id]['dist_coeffs']
            rvec = camera_params[cam_id]['rvecs'][frame_id]
            tvec = camera_params[cam_id]['tvecs'][frame_id]
            objp = objpoints_dict[cam_id][frame_id]
            imgp = imgpoints_dict[cam_id][frame_id]

            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            # proj = project_points_numpy(objp, rvec, tvec, K, dist)
            proj = proj.squeeze()
            imgp = imgp.squeeze()

            for pt in proj:
                cv2.circle(img, tuple(int(x) for x in pt), 5, (0, 255, 0), 2)  # green: reprojected
            for pt in imgp:
                cv2.circle(img, tuple(int(x) for x in pt), 3, (0, 0, 255), 1)  # red: detected

            cv2.resize(img, (360, 360), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(out_dir, img_name), img)

# def filter_points_with_ransac_pnp(camera_params, objpoints_dict, imgpoints_dict, threshold=4.0):
#     """
#     Use cv2.solvePnPRansac to detect and remove 2D-3D outlier matches in each frame.
#     Updates objpoints_dict and imgpoints_dict in-place.
#     Also logs per-camera average reprojection error before and after filtering.
#     """
#     for cam_id in objpoints_dict:
#         K = camera_params[cam_id]['camera_matrix']
#         dist = camera_params[cam_id]['dist_coeffs']
#         init_errors = []
#         final_errors = []
#         for frame_id in list(objpoints_dict[cam_id].keys()):
#             objp = objpoints_dict[cam_id][frame_id]
#             imgp = imgpoints_dict[cam_id][frame_id]

#             if objp.shape[0] < 6:
#                 continue

#             # Compute initial error
#             proj_init, _ = cv2.projectPoints(objp, camera_params[cam_id]['rvecs'][frame_id],
#                                              camera_params[cam_id]['tvecs'][frame_id], K, dist)
#             err_init = np.linalg.norm(proj_init.squeeze() - imgp.squeeze(), axis=1)
#             init_errors.extend(err_init.tolist())

#             success, rvec, tvec, inliers = cv2.solvePnPRansac(
#                 objp, imgp, K, dist,
#                 reprojectionError=threshold,
#                 confidence=0.99,
#                 flags=cv2.SOLVEPNP_ITERATIVE)

#             if not success or inliers is None or len(inliers) < 10:
#                 print(f"[RANSAC] Removing frame {cam_id}-{frame_id} (insufficient inliers)")
#                 del objpoints_dict[cam_id][frame_id]
#                 del imgpoints_dict[cam_id][frame_id]
#                 continue

#             inliers = inliers.flatten()
#             objp_filtered = objp[inliers]
#             imgp_filtered = imgp[inliers]

#             # Recompute error after filtering
#             proj_final, _ = cv2.projectPoints(objp_filtered, rvec, tvec, K, dist)
#             err_final = np.linalg.norm(proj_final.squeeze() - imgp_filtered.squeeze(), axis=1)
#             final_errors.extend(err_final.tolist())

#             objpoints_dict[cam_id][frame_id] = objp_filtered
#             imgpoints_dict[cam_id][frame_id] = imgp_filtered

#             # update pose with inlier-based refinement
#             camera_params[cam_id]['rvecs'][frame_id] = rvec
#             camera_params[cam_id]['tvecs'][frame_id] = tvec

#         if init_errors and final_errors:
#             print(f"[RANSAC] Camera {cam_id}: Mean error before = {np.mean(init_errors):.2f} px, after = {np.mean(final_errors):.2f} px")
#     return camera_params

# def evaluate_camera_pairwise_reprojection(camera_params, extrinsics, objpoints, imgpoints, image_dir):
#     os.makedirs("pairwise_debug", exist_ok=True)

#     for (camA, camB), (R, T) in extrinsics.items():
#         print(f"[Eval] {camA} → {camB}")

#         K_A = camera_params[camA]['camera_matrix']
#         D_A = camera_params[camA]['dist_coeffs']
#         K_B = camera_params[camB]['camera_matrix']
#         D_B = camera_params[camB]['dist_coeffs']

#         shared_frames = set(objpoints[camA].keys()) & set(objpoints[camB].keys())
#         total_error = 0
#         total_points = 0

#         for frame_id in sorted(shared_frames):
#             obj = objpoints[camA][frame_id]          # world (board) points
#             imgB = imgpoints[camB][frame_id]         # camB detected corners

#             ##############################################
#             # Compute initial error
#             objA = objpoints[camA][frame_id]          # world (board) points
#             objA = objA @ cv2.Rodrigues(camera_params[camA]['rvecs'][frame_id])[0].T + camera_params[camA]['tvecs'][frame_id].reshape(1, 3)
#             objB = objA @ R.T + T.reshape(1, 3)  # transform to camB frame
#             objB = objB @ cv2.Rodrigues(camera_params[camB]['rvecs'][frame_id])[0].T + camera_params[camB]['tvecs'][frame_id].reshape(1, 3)
#             imgB = imgpoints[camB][frame_id]         # camB detected corners
#             projB, _ = cv2.projectPoints(objB, np.eye(3), np.zeros((3,1)), K_B, D_B)

#             objB = objpoints[camB][frame_id]          # world (board) points
#             imgB = imgpoints[camB][frame_id]         # camB detected corners
#             projB, _ = cv2.projectPoints(objB, camera_params[camB]['rvecs'][frame_id],
#                                              camera_params[camB]['tvecs'][frame_id], K_B, D_B)


#             vis = cv2.imread(f"data2/{camB}_{frame_id}.png")
#             for pt in projB:
#                 cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 3, (0, 255, 0), -1)  # 투영점
#             for pt in imgB:
#                 cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 2, (0, 0, 255), 1)   # 실제 감지점

#             save_path = os.path.join("pairwise_debug", f"{camA}_{camB}_{frame_id}.png")
#             cv2.imwrite(save_path, vis)
        


#             # Compute initial error
#             objA = objpoints[camA][frame_id]          # world (board) points
#             objA = objA @ cv2.Rodrigues(camera_params[camA]['rvecs'][frame_id])[0].T + camera_params[camA]['tvecs'][frame_id].reshape(1, 3)  # transform to camB frame
#             imgA = imgpoints[camA][frame_id]         # camB detected corners
#             projA, _ = cv2.projectPoints(objA, np.eye(3), np.zeros((3, 1)), K_A, D_A)

#             vis = cv2.imread(f"data2/{camA}_{frame_id}.png")
#             for pt in projA:
#                 cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 3, (0, 255, 0), -1)  # 투영점
#             for pt in imgA:
#                 cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 2, (0, 0, 255), 1)   # 실제 감지점

#             save_path = os.path.join("pairwise_debug", f"{camA}_{camA}_{frame_id}.png")
#             cv2.imwrite(save_path, vis)
        

#             # Compute initial error
#             objA = objpoints[camA][frame_id]          # world (board) points
#             imgA = imgpoints[camA][frame_id]         # camB detected corners
#             projA, _ = cv2.projectPoints(objA, camera_params[camA]['rvecs'][frame_id],
#                                              camera_params[camA]['tvecs'][frame_id], K_A, D_A)

#             vis = cv2.imread(f"data2/{camA}_{frame_id}.png")
#             for pt in projA:
#                 cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 3, (0, 255, 0), -1)  # 투영점
#             for pt in imgA:
#                 cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 2, (0, 0, 255), 1)   # 실제 감지점

#             save_path = os.path.join("pairwise_debug", f"{camA}_{camA}_{frame_id}2.png")
#             cv2.imwrite(save_path, vis)
        


#             # Compute initial error
#             objB = objpoints[camB][frame_id]          # world (board) points
#             projB, _ = cv2.projectPoints(objB, camera_params[camB]['rvecs'][frame_id],
#                                              camera_params[camB]['tvecs'][frame_id], K_B, D_B)

#             vis = cv2.imread(f"data2/{camB}_{frame_id}.png")
#             for pt in projB:
#                 cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 3, (0, 255, 0), -1)  # 투영점
#             for pt in imgB:
#                 cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 2, (0, 0, 255), 1)   # 실제 감지점

#             save_path = os.path.join("pairwise_debug", f"{camA}_{camB}_{frame_id}.png")
#             cv2.imwrite(save_path, vis)
        
#             ##############################################

#             # camA 기준 보드 좌표를 camB에 투영 (extrinsic: camA→camB)
#             rvec, _ = cv2.Rodrigues(R)
#             tvec = T.reshape(3, 1)

#             proj, _ = cv2.projectPoints(obj, rvec, tvec, K_B, D_B)
#             proj = proj.reshape(-1, 2)

#             error = np.linalg.norm(proj - imgB, axis=1)
#             total_error += np.sum(error)
#             total_points += len(error)

#             try:
#                 # 시각화
#                 vis = cv2.imread(f"data2/{camB}_{frame_id}.png")
#                 for pt in proj:
#                     cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 3, (0, 255, 0), -1)  # 투영점
#                 for pt in imgB:
#                     cv2.circle(vis, tuple(np.round(pt.squeeze()).astype(int)), 2, (0, 0, 255), 1)   # 실제 감지점

#                 save_path = os.path.join("pairwise_debug", f"{camA}_{camB}_{frame_id}.png")
#                 cv2.imwrite(save_path, vis)
#             except:
#                 continue

#         if total_points > 0:
#             mean_error = total_error / total_points
#             print(f"  → Mean reprojection error: {mean_error:.2f} px")
#         else:
#             print(f"  → No shared frames to evaluate.")
