import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def create_camera_mesh(scale=1.0):
    """Create a simple Open3D pyramid mesh to represent a camera."""
    points = np.array([
        [0, 0, 0],
        [1, 0.75, 1.0],
        [1, -0.75, 1.0],
        [-1, -0.75, 1.0],
        [-1, 0.75, 1.0]
    ]) * scale
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    colors = [[0, 0, 1] for _ in lines]  # blue
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def draw_cameras_and_points(camera_params, extrinsics=None, board_points=None, board_shape=(9, 6), ref_cam='00'):
    """
    Visualize camera poses and optional calibration board in 3D.

    Parameters:
        camera_params (dict): Output of calibrate_individual_cameras
        extrinsics (dict): Relative poses between cameras
        board_points (np.ndarray): Optional Nx3 array of calibration board
    """
    vis_list = []

    origin = np.eye(4)

    for cam_id in camera_params:
        cam = create_camera_mesh(scale=30.0)
        T = np.eye(4)

        if cam_id == ref_cam:
            T = origin
        else:
            R, t = extrinsics[cam_id]
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

        cam.transform(T)
        vis_list.append(cam)

    boardp = get_board_points_in_world(board_points, camera_params, ref_cam='00')
    color_list = get_distinct_colors(len(board_points[ref_cam]))

    for i, (frame_id, pts) in enumerate(boardp.items()):
        if pts.shape[0] != board_shape[0] * board_shape[1]:
            continue

        line_board = create_board_lines(pts, board_shape, color_list[i])
        vis_list.append(line_board)

    o3d.visualization.draw_geometries(vis_list)


def create_board_lines(points, board_shape=(9, 6), color=[0, 1, 0]):
    """
    points: (N, 3) ndarray
    board_shape: (cols, rows) of inner corners (e.g., (9, 6))
    return: open3d.geometry.LineSet
    """
    cols, rows = board_shape
    lines = []

    # connect each row's points
    for r in range(rows):
        for c in range(cols - 1):
            i1 = r * cols + c
            i2 = r * cols + (c + 1)
            lines.append([i1, i2])

    # connect each column's points
    for c in range(cols):
        for r in range(rows - 1):
            i1 = r * cols + c
            i2 = (r + 1) * cols + c
            lines.append([i1, i2])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines)) 

    return line_set

def get_distinct_colors(n):
    cmap = plt.get_cmap("tab20")  # 20 distinct colors
    colors = [cmap(i % 20)[:3] for i in range(n)]
    return colors



def get_board_points_in_world(objpoints_dict, camera_params, ref_cam='00'):
    """
    Transform all detected board points from camera coordinates
    to world coordinates using a reference camera.

    Parameters:
        objpoints_dict (dict): 3D object points [cam][frame] = (N, 3)
        camera_params (dict): Calibration results including rvecs/tvecs
        ref_cam (str): Camera ID to use as world reference

    Returns:
        board_world_points (dict): {frame_id: (N, 3)} board points in world coordinates
    """
    board_world_points = {}

    for frame_id in objpoints_dict[ref_cam]:
        board_pts = objpoints_dict[ref_cam][frame_id]  # (N, 3)

        rvec = camera_params[ref_cam]['rvecs'][frame_id]
        tvec = camera_params[ref_cam]['tvecs'][frame_id]
        R, _ = cv2.Rodrigues(rvec)

        world_pts = (R @ board_pts.T + tvec).T  # (N, 3)
        board_world_points[frame_id] = world_pts

    return board_world_points


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