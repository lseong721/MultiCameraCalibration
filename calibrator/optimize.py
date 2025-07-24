import numpy as np
import cv2
from scipy.optimize import least_squares

def pack_params(camera_params, extrinsics, optimize_intrinsic=True):
    """
    Pack parameters into a 1D vector.
    - camera intrinsics (fx, fy, cx, cy, dist) if optimize_intrinsic
    - extrinsics (R, t)
    """
    params = []
    cam_intrinsic_map = {}
    extrinsic_map = {}

    for idx, cam_id in enumerate(camera_params):
        if optimize_intrinsic:
            K = camera_params[cam_id]['camera_matrix']
            dist = camera_params[cam_id]['dist_coeffs'].flatten()[:5]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            cam_intrinsic_map[cam_id] = len(params)
            params.extend([fx, fy, cx, cy])
            params.extend(dist)

    for cam_id, (R, t) in extrinsics.items():
        if cam_id == '00':  # ref_cam
            continue
        rvec, _ = cv2.Rodrigues(R)
        extrinsic_map[cam_id] = len(params)
        params.extend(rvec.flatten())
        params.extend(t.flatten())

    return np.array(params), cam_intrinsic_map, extrinsic_map

def unpack_params(params, camera_params, cam_intrinsic_map, extrinsic_map, optimize_intrinsic):
    new_intrinsics = {}
    new_extrinsics = {}

    for cam_id in camera_params:
        if optimize_intrinsic:
            start = cam_intrinsic_map[cam_id]
            fx, fy, cx, cy = params[start:start+4]
            dist = params[start+4:start+9]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        else:
            K = camera_params[cam_id]['camera_matrix']
            dist = camera_params[cam_id]['dist_coeffs'].flatten()[:5]
        new_intrinsics[cam_id] = (K, dist)

    for cam_id in camera_params:
        if cam_id == '00':
            new_extrinsics[cam_id] = (np.eye(3), np.zeros((3, 1)))
        else:
            start = extrinsic_map[cam_id]
            rvec = params[start:start+3].reshape(3, 1)
            tvec = params[start+3:start+6].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            new_extrinsics[cam_id] = (R, tvec)

    return new_intrinsics, new_extrinsics

def reprojection_error(params, camera_params, objpoints, imgpoints,
                       cam_intrinsic_map, extrinsic_map, optimize_intrinsic):
    errors = []
    new_intrinsics, new_extrinsics = unpack_params(params, camera_params, cam_intrinsic_map, extrinsic_map, optimize_intrinsic)

    for cam_id in camera_params:
        if cam_id not in objpoints:
            continue
        K, dist = new_intrinsics[cam_id]
        R, t = new_extrinsics[cam_id]

        for frame_id in objpoints[cam_id]:
            objp = objpoints[cam_id][frame_id]
            imgp = imgpoints[cam_id][frame_id]

            try:
                R_A, _ = cv2.Rodrigues(camera_params['00']['rvecs'][frame_id])
                t_A = camera_params['00']['tvecs'][frame_id]
            except:
                continue

            objA = objp @ R_A.T + t_A.reshape(1, 3)
            objA2B = objA @ R.T + t.reshape(1, 3)

            proj, _ = cv2.projectPoints(objA2B, np.zeros((3,1)), np.zeros((3,1)), K, dist)
            error = np.linalg.norm(proj.squeeze() - imgp.squeeze(), axis=1)
            errors.append(error.ravel())

    return np.concatenate(errors)

def optimize_bundle(camera_params, extrinsics, objpoints, imgpoints, optimize_intrinsic=True):
    x0, cam_intrinsic_map, extrinsic_map = pack_params(
        camera_params, extrinsics, optimize_intrinsic)
    
    init_errors = reprojection_error(x0, camera_params, objpoints, imgpoints, cam_intrinsic_map, extrinsic_map, optimize_intrinsic)

    result = least_squares(
        reprojection_error,
        x0,
        verbose=2,
        method='trf',
        max_nfev=200,
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        args=(camera_params, objpoints, imgpoints, cam_intrinsic_map, extrinsic_map, optimize_intrinsic),
    )

    optimized_intrinsics, optimized_extrinsics = unpack_params(
        result.x, camera_params, cam_intrinsic_map, extrinsic_map, optimize_intrinsic)

    final_errors = reprojection_error(result.x, camera_params, objpoints, imgpoints, cam_intrinsic_map, extrinsic_map, optimize_intrinsic)

    for cam_id in camera_params:
        camera_params[cam_id]['camera_matrix'] = optimized_intrinsics[cam_id][0]
        camera_params[cam_id]['dist_coeffs'] = optimized_intrinsics[cam_id][1]

    print(f"[BA] Mean error before = {np.mean(init_errors):.2f} px, after = {np.mean(final_errors):.2f} px")

    return result, optimized_extrinsics
