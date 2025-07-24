import open3d as o3d
import numpy as np

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

def draw_cameras_and_points(camera_params, extrinsics=None, board_points=None, ref_cam='00'):
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
        cam = create_camera_mesh(scale=10.0)
        T = np.eye(4)

        if cam_id == ref_cam:
            T = origin
        else:
            R, t = extrinsics[cam_id]
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

        cam.transform(T)
        vis_list.append(cam)

    if board_points is not None:
        board = o3d.geometry.PointCloud()
        board.points = o3d.utility.Vector3dVector(board_points)
        board.paint_uniform_color([1, 0, 0])  # red
        vis_list.append(board)

    o3d.visualization.draw_geometries(vis_list)
