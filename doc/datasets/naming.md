# 命名约定

## 数据集名

- `human2robot`：外部原始配对数据集名。
- `G1`：Unitree G1 robot dataset / WBT / MainCamOnly 数据。

## 数据方向

- `h2r`：human -> robot。
- `r2h`：robot -> human。
- `identity_r2r`：robot -> robot identity。
- `blur_r2r`：blurred robot -> clear robot。

## duration

- `1s`：旧 1 秒数据。
- `2s61f30`：2 秒语义、61 帧、30fps。
- `2s61f30_slide`：2 秒 Seedance 滑窗。
- `2s61f30_human2robot_v1`：human2robot MP4 源。
- `2s61f30_human2robot_hdf5_v1`：human2robot HDF5 源。

## 规则

- 新的 human2robot 派生产物必须在 duration 中写 `human2robot`，不要写成
  `2s61f30_h2r_v1`。
- `h2r` / `r2h` 只表示训练方向或 `data_type`。
- `video/` 与 `control_video/` 的语义以当前 `data_type` 和 manifest 中
  `input_role` / `target_role` 为准。

