# Multi-Camera Calibration and Bundle Adjustment

This project performs multi-camera calibration using chessboard detection, followed by bundle adjustment to refine intrinsic and extrinsic parameters.

## 📁 Dataset Structure

All calibration images should be placed in the `data/` directory.

### 📌 File Naming Convention

Each image file should follow the naming pattern:

```
data/{camera_id}_{frame_id}.png
```

- `camera_id`: a two-digit string representing the camera index (e.g., `00`, `01`, `02`, ...).
- `frame_id`: a six-digit string representing the frame index (e.g., `000000`, `000001`, ...).

### ✅ Example Files

```
data/
├── 00_000000.png
├── 00_000001.png
├── 01_000000.png
├── 01_000001.png
├── ...
```

These images should all contain the same chessboard pattern for reliable multi-camera calibration.

## ⚙️ Calibration Requirements

- All cameras should observe the same chessboard in a sufficient number of frames.
- Shared frames (same `frame_id`) across different `camera_id`s are used to compute relative poses.

## 📦 Outputa

- Camera intrinsics and extrinsics are saved or printed after bundle adjustment.
- Optional reprojection visualizations and error reports are also generated.

## 🛠️ Dependencies

- Python 3.8+
- OpenCV
- NumPy
- SciPy

## 🖼️ Example Result

Example visualization of calibrated camera poses and board positions:

![Result](figs/result.png)
