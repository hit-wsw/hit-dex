import numpy as np


between_cam_3 = np.eye(4)
between_cam_3[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]])
between_cam_3[:3, 3] = np.array([0.0, -0.064, 0.0])

pose_path = '/media/wsw/SSD1T1/data/save_wipe_1-14/save_data_wipe_1-14_01/frame_70/pose_3.txt'
pose_3 = np.loadtxt(pose_path)



right_hand_joint_xyz = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
])

right_joint_xyz_reshaped = right_hand_joint_xyz[:, :, np.newaxis]
print(right_joint_xyz_reshaped[:, :, 0])  # 输出: (5, 3, 1)

#print((pose_3 @ between_cam_3)-(between_cam_3 @ pose_3))
