import numpy as np
import cv2
from itertools import combinations
from collections import defaultdict, deque
import os

def compute_relative_pose(R_ref, t_ref, R_other, t_other):
    R = R_other @ R_ref.T
    t = -R @ t_ref + t_other
    return R, t

def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t.reshape(3, 1)
    return T

def decompose_transform(T):
    R = T[:3, :3]
    t = T[:3, 3:]
    return R, t

def estimate_pairwise_extrinsics(camera_params):
    pairwise_extrinsics = {}
    cams = list(camera_params.keys())

    for camA, camB in combinations(cams, 2):
        shared_frames = (
            set(camera_params[camA]['rvecs'].keys()) &
            set(camera_params[camB]['rvecs'].keys())
        )
        if not shared_frames:
            continue

        transforms = []

        for frame_id in shared_frames:
            R_A, _ = cv2.Rodrigues(camera_params[camA]['rvecs'][frame_id])
            t_A = camera_params[camA]['tvecs'][frame_id]
            R_B, _ = cv2.Rodrigues(camera_params[camB]['rvecs'][frame_id])
            t_B = camera_params[camB]['tvecs'][frame_id]

            R_AB, t_AB = compute_relative_pose(R_A, t_A, R_B, t_B)
            T_AB = make_transform(R_AB, t_AB)

            transforms.append(T_AB)

        T_avg = np.mean(transforms, axis=0)
        R_avg, t_avg = decompose_transform(T_avg)
        pairwise_extrinsics[(camA, camB)] = (R_avg, t_avg)
        pairwise_extrinsics[(camB, camA)] = (R_avg.T, -R_avg.T @ t_avg)

    return pairwise_extrinsics

def build_camera_graph(pairwise_extrinsics):
    graph = defaultdict(list)
    for (camA, camB), (R, t) in pairwise_extrinsics.items():
        graph[camA].append((camB, R, t))
    return graph

def traverse_paths(graph, ref_cam):
    # BFS 경로 기록 (shortest first)
    paths = {}
    queue = deque([[ref_cam]])
    visited = set()

    while queue:
        path = queue.popleft()
        last = path[-1]
        if last in visited:
            continue
        visited.add(last)
        paths[last] = path
        for neighbor, _, _ in graph[last]:
            if neighbor not in path:
                queue.append(path + [neighbor])
    return paths

def evaluate_reprojection_error(R, T, ref_cam, cam_id, frame_id, camera_params, objpoints_dict, imgpoints_dict):
    objp = objpoints_dict[cam_id][frame_id]
    imgp = imgpoints_dict[cam_id][frame_id]
    K = camera_params[cam_id]['camera_matrix']
    D = camera_params[cam_id]['dist_coeffs']

    R_A, _ = cv2.Rodrigues(camera_params[ref_cam]['rvecs'][frame_id])
    t_A = camera_params[ref_cam]['tvecs'][frame_id]

    objA = objp @ R_A.T + t_A.reshape(1, 3)
    objA2B = objA @ R.T + T.reshape(1, 3)
    proj, _ = cv2.projectPoints(objA2B, np.zeros((3,1)), np.zeros((3,1)), K, D)
    proj = proj.squeeze()

    # # # === image save for debug ===
    # vis_imgA = cv2.imread(f"D:/Workspace/Dataset/Multi-camera calibration/selected_images_2400/resize_720/{cam_id}_{frame_id}.png")

    # proj, _ = cv2.projectPoints(objA2B, np.zeros((3,1)), np.zeros((3,1)), K, D)

    # for pt in proj.reshape(-1, 2):
    #     cv2.circle(vis_imgA, tuple(np.round(pt.squeeze()).astype(np.int32)), 3, (0, 255, 0), -1)
    # for pt in imgp:
    #     cv2.circle(vis_imgA, tuple(np.round(pt.squeeze()).astype(np.int32)), 2, (0, 0, 255), 1)

    # cv2.imwrite(f"pairwise_debug/{cam_id}_{frame_id}_A.png", vis_imgA)   

    return np.linalg.norm(proj - imgp.squeeze(), axis=1).mean()

def estimate_extrinsics_via_best_path(camera_params, objpoints_dict, imgpoints_dict, ref_cam='00'):
    pairwise_extrinsics = estimate_pairwise_extrinsics(camera_params)
    graph = build_camera_graph(pairwise_extrinsics)
    cam_paths = traverse_paths(graph, ref_cam)

    extrinsics = {ref_cam: (np.eye(3), np.zeros((3, 1)))}

    for cam_id, path in cam_paths.items():
        if cam_id == ref_cam:
            continue

        best_error = np.inf
        best_R, best_t = None, None

        # cumulative transform
        for frame_id in sorted(set(camera_params[cam_id]['rvecs'].keys())):
            try:
                for i in range(len(path) - 1):
                    A, B = path[i], path[i+1]
                    R, t = pairwise_extrinsics[(A, B)]

                error = evaluate_reprojection_error(R, t, ref_cam, cam_id, frame_id, camera_params, objpoints_dict, imgpoints_dict)
                if error < best_error:
                    best_error = error
                    best_R = R
                    best_t = t
            except:
                continue

        if best_R is not None:
            R_final = best_R
            t_final = best_t
            extrinsics[cam_id] = (R_final, t_final)
            print(f"[✓] {ref_cam} → {cam_id} via {path}: error={best_error:.2f}")
        else:
            print(f"[X] Failed to compute extrinsic for {ref_cam} → {cam_id}")


    return extrinsics