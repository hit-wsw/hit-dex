import open3d as o3d
import numpy as np

num_points_to_sample = 10000
d435i_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    1280, 720,
    923.995567,
    923.492024,
    620.892371,
    384.178306)

head_pose = np.eye(4)
head_pose[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
head_pose[:3, 3] = np.array([0.0, 0.0, 0.0])

# process pointcloud
color_image_o3d = o3d.io.read_image(color_jpg_path)
depth_image_o3d = o3d.io.read_image(depth_png_path)
max_depth = 1
depth_array = np.asarray(depth_image_o3d)
mask = depth_array > max_depth
depth_array[mask] = 0
filtered_depth_image = o3d.geometry.Image(depth_array)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, filtered_depth_image, depth_trunc=4.0, convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, d435i_depth_intrinsic)
pcd.transform(head_pose)
color_pcd = np.concatenate((np.array(pcd.points), np.array(pcd.colors)), axis=-1)

# down sample input pointcloud
if len(color_pcd) > num_points_to_sample:
    indices = np.random.choice(len(color_pcd), num_points_to_sample, replace=False)
    color_pcd = color_pcd[indices]

    

pointcloud = self.pointcloud_data[(self.current_index-self.obs_horizon):self.current_index]
waist = self.waist_data[(self.current_index-self.obs_horizon):self.current_index]
robot_arm = self.robot_arm_data[(self.current_index-self.obs_horizon):self.current_index]
robot_hand = self.robot_hand_data[(self.current_index-self.obs_horizon):self.current_index]

# postprocess data frames
#agentview_image = ObsUtils.batch_image_hwc_to_chw(agentview_image) / 255.0
#agentview_image = TensorUtils.to_device(torch.FloatTensor(agentview_image), self.device)

pointcloud = TensorUtils.to_device(torch.FloatTensor(pointcloud), self.device)
waist = TensorUtils.to_device(torch.FloatTensor(waist), self.device)
robot_arm = TensorUtils.to_device(torch.FloatTensor(robot_arm), self.device)
robot_hand = TensorUtils.to_device(torch.FloatTensor(robot_hand), self.device)

return_state = {
    'pointcloud': pointcloud,
    'waist': waist,
    'robot_arm': robot_arm,
    'robot_hand': robot_hand,
}