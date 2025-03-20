# ffmpeg -framerate 10 -i saved_current_frames/%d.jpg -c:v mpeg4 -pix_fmt yuv420p saved_run_video.mp4

import argparse
import os
import json
import h5py
import imageio
import sys
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2
import re
import torch
import open3d as o3d
from transforms3d.euler import quat2mat
from step2_pybullet_ik_bimanual import LeapPybulletIK
from step2_utils import *

from glob import glob
from threading import Thread, Lock
from run_trained_agent_utils import _back_project_point, apply_pose_matrix
from transforms3d.quaternions import mat2quat
from scipy.spatial.transform import Rotation
from transforms3d.axangles import axangle2mat

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class VisOnlyEnv:
    def __init__(self, hdf5_file, obs_horizon = 3):
        self.hdf5_file = hdf5_file
        self.lock = Lock()
        self.running = False
        self.current_image = None
        self.current_image_id = 0
        self.obs_horizon = obs_horizon
        self.current_index = self.obs_horizon

        self.agentview_image_data = None
        self.pointcloud_data = None
        self.robot0_eef_pos_data = None
        self.robot0_eef_quat_data = None
        self.robot0_eef_hand_data = None
        self.robot0_eef = None

        self.action_data = None
        self.done_data = None
        self.state_data = None

        self.demo_index = 0
        self.sorted_demos = None
        self.load_hdf5_data()

        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    def load_hdf5_data(self):
        with h5py.File(self.hdf5_file, 'r') as file:
            data_group = file['data']
            self.sorted_demos = sorted(data_group.keys(), key=lambda x: int(x.split('_')[-1]))
            self._load_demo(self.sorted_demos[0])

    def _load_demo(self, demo_name):
        with h5py.File(self.hdf5_file, 'r') as file:
            demo_group = file['data'][demo_name]

            self.agentview_image_data = demo_group['obs']['agentview_image'][:]

            self.pointcloud_data = demo_group['obs']['pointcloud'][:]
            self.robot0_eef_pos_data = demo_group['obs']['robot0_eef_pos'][:]
            self.robot0_eef_quat_data = demo_group['obs']['robot0_eef_quat'][:]
            self.robot0_eef_hand_data = demo_group['obs']['robot0_eef_hand'][:]
            self.robot0_eef = np.concatenate((self.robot0_eef_pos_data, self.robot0_eef_quat_data, self.robot0_eef_hand_data), axis=-1)

            self.action_data = demo_group['actions'][:]
            self.done_data = demo_group['dones'][:]
            self.state_data = demo_group['states'][:]

            self.current_index = self.obs_horizon

    def get_state(self):
        with self.lock:
            if self.current_index < len(self.agentview_image_data):
                agentview_image = self.agentview_image_data[(self.current_index-self.obs_horizon):self.current_index]

                self.current_image = agentview_image[-1]
                #self.pointcloud_data = self.pointcloud_data[0]
                pointcloud = self.pointcloud_data[(self.current_index-self.obs_horizon):self.current_index]
                robot0_eef_pos = self.robot0_eef_pos_data[(self.current_index-self.obs_horizon):self.current_index]
                robot0_eef_quat = self.robot0_eef_quat_data[(self.current_index-self.obs_horizon):self.current_index]
                robot0_eef_hand = self.robot0_eef_hand_data[(self.current_index-self.obs_horizon):self.current_index]

                # postprocess data frames
                #agentview_image = ObsUtils.batch_image_hwc_to_chw(agentview_image) / 255.0
                #agentview_image = TensorUtils.to_device(torch.FloatTensor(agentview_image), self.device)

                pointcloud = TensorUtils.to_device(torch.FloatTensor(pointcloud), self.device)
                robot0_eef_pos = TensorUtils.to_device(torch.FloatTensor(robot0_eef_pos), self.device)
                robot0_eef_quat = TensorUtils.to_device(torch.FloatTensor(robot0_eef_quat), self.device)
                robot0_eef_hand = TensorUtils.to_device(torch.FloatTensor(robot0_eef_hand), self.device)

                return_state = {
                    'pointcloud': pointcloud,
                    'robot0_eef_pos': robot0_eef_pos,
                    'robot0_eef_quat': robot0_eef_quat,
                    'robot0_eef_hand': robot0_eef_hand,
                }

                done = bool(self.done_data[self.current_index])
                self.current_index += 1
                return return_state, done
            else:
                return None, True  # Return True for done if no more images
            

    def show_pcd(self):
        i = 25
        color_pcd = self.pointcloud_data[i]
        pos_right = self.robot0_eef_pos_data[i][:3]
        ori_right = self.robot0_eef_quat_data[i][:4]
        hand_ori = quat2mat(ori_right)
        

        pos_left = self.robot0_eef_pos_data[i][3:]
        ori_left = self.robot0_eef_quat_data[i][4:]
        hand_ori1 = quat2mat(ori_left)
        

        
        
        # 计算右手的位姿
        '''pose_right = np.eye(4)
        pose_right[:3, :3] = quat2mat(self.robot0_eef_quat_data[i][:4])
        pose_right[:3, 3] = self.robot0_eef_pos_data[i][:3]
        mat = np.array([[0, -1, 0],[0, 0, 1],[-1, 0, 0]])
        

        update_pose_3 = robot_to_hand(pose_right)
        hand_ori = update_pose_3[:3, :3]  # 右手坐标系的旋转矩阵
        offset = np.array([[-0.02, 0.05, -0.1]])
        offset = offset @ hand_ori.T
        pos_right = update_pose_3[:3, 3].reshape(1, 3) + offset
        pos_right = pos_right.reshape(3, 1)
        hand_ori = hand_ori @ mat

        # 计算左手的位姿
        pose_left = np.eye(4)
        pose_left[:3, :3] = quat2mat(self.robot0_eef_quat_data[i][4:])
        pose_left[:3, 3] = self.robot0_eef_pos_data[i][3:]

        update_pose_2 = robot_to_hand_left(pose_left)
        hand_ori1 = update_pose_2[:3, :3]  # 左手坐标系的旋转矩阵
        offset1 = np.array([[0.02, 0.05, -0.1]])
        offset1 = offset1 @ hand_ori1.T
        pos_left = update_pose_2[:3, 3].reshape(1, 3) + offset1
        pos_left = pos_left.reshape(3, 1)
        hand_ori1 = hand_ori1 @ mat
        update_pose_2[:3, 3] = pos_left.flatten()'''

        # 将颜色范围放大到 [0, 255]
        color_pcd[:, 3:] *= 255

        # 创建 3D 图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])  # 设置比例尺一致

        # 提取坐标和颜色
        points = color_pcd[:, :3]  # 点的坐标 (x, y, z)
        colors = color_pcd[:, 3:] / 255.0  # 点的颜色 (r, g, b)，归一化到 [0, 1]

        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=1)
        ax.scatter(pos_right[0], pos_right[1], pos_right[2], c='blue', label='Right End-Effector')
        ax.scatter(pos_left[0], pos_left[1], pos_left[2], c='red', label='Left End-Effector')
        ax.scatter(0, 0, 0)

        # 绘制右手坐标系
        axis_length = 0.1  # 坐标轴长度
        ax.quiver(pos_right[0], pos_right[1], pos_right[2], 
                hand_ori[0, 0], hand_ori[1, 0], hand_ori[2, 0], 
                color='r', length=axis_length, normalize=True, label='Right X')
        ax.quiver(pos_right[0], pos_right[1], pos_right[2], 
                hand_ori[0, 1], hand_ori[1, 1], hand_ori[2, 1], 
                color='g', length=axis_length, normalize=True, label='Right Y')
        ax.quiver(pos_right[0], pos_right[1], pos_right[2], 
                hand_ori[0, 2], hand_ori[1, 2], hand_ori[2, 2], 
                color='b', length=axis_length, normalize=True, label='Right Z')

        # 绘制左手坐标系
        ax.quiver(pos_left[0], pos_left[1], pos_left[2], 
                hand_ori1[0, 0], hand_ori1[1, 0], hand_ori1[2, 0], 
                color='r', length=axis_length, normalize=True, label='Left X')
        ax.quiver(pos_left[0], pos_left[1], pos_left[2], 
                hand_ori1[0, 1], hand_ori1[1, 1], hand_ori1[2, 1], 
                color='g', length=axis_length, normalize=True, label='Left Y')
        ax.quiver(pos_left[0], pos_left[1], pos_left[2], 
                hand_ori1[0, 2], hand_ori1[1, 2], hand_ori1[2, 2], 
                color='b', length=axis_length, normalize=True, label='Left Z')

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置图形标题
        ax.set_title(f'3D Point Cloud Visualization - Frame {i}')

        # 显示图例
        ax.legend()

        # 手动设置坐标轴范围，确保比例尺一致
        max_range = np.array([points[:, 0].max()-points[:, 0].min(), 
                            points[:, 1].max()-points[:, 1].min(), 
                            points[:, 2].max()-points[:, 2].min()]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 显示图形
        plt.show()
        # plt.pause(0.1)  # 暂停0.1秒，以便观察每一帧的变化
        # plt.close()  # 关闭当前图形，准备绘制下一帧

    def show_pos(self):
        # 创建 3D 图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 初始化空的末端执行器位置
        right_eef, = ax.plot([], [], [], 'bo', label='Right End-Effector')
        left_eef, = ax.plot([], [], [], 'ro', label='Left End-Effector')

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置图形标题
        ax.set_title('End-Effector Position Visualization')

        # 显示图例
        ax.legend()

        # 设置坐标轴范围（根据数据动态调整）
        ax.set_xlim([0, 0.6])
        ax.set_ylim([-0.6, 0.4])
        ax.set_zlim([-0.4, 0.4])
        ax.scatter(0, 0, 0, color='green', label='Origin')

        def update(frame):
            # 获取当前帧的数据
            pos_right = self.robot0_eef_pos_data[frame][:3]
            pos_left = self.robot0_eef_pos_data[frame][3:]

            # 更新末端执行器位置
            right_eef.set_data(pos_right[0], pos_right[1])  # 更新 X 和 Y
            right_eef.set_3d_properties(pos_right[2])      # 更新 Z
            left_eef.set_data(pos_left[0], pos_left[1])    # 更新 X 和 Y
            left_eef.set_3d_properties(pos_left[2])        # 更新 Z

            # 更新标题
            ax.set_title(f'End-Effector Position Visualization - Frame {frame}')

            return right_eef, left_eef

        # 创建动画
        ani = FuncAnimation(fig, update, frames=range(100), interval=100, blit=True)

        # 显示图形
        plt.show()

    def vis_model_action(self, action):
        self.current_image = self._draw_action(self.current_image, self.robot0_eef[self.current_index - 2], self.state_data[self.current_index - 2], action, gt=False)
    def vis_gt_action(self):
        self.current_image = self._draw_action(self.current_image, self.robot0_eef[self.current_index - 2], self.state_data[self.current_index - 2], self.action_data[self.current_index - 2])
    def vis_gt_hand(self,hand_ee):
        self.current_image = self._draw_ee_hand(self.current_image, hand_ee, self.state_data[self.current_index - 2])

    def save_current_frame(self):
        output_filename = '/home/wsw/Dexcap/pic/{}.jpg'.format(self.current_image_id)
        cv2.imwrite(output_filename, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
        self.current_image_id += 1


    def reset(self):
        with self.lock:
            self.demo_index = (self.demo_index + 1) % len(self.sorted_demos)
            self._load_demo(self.sorted_demos[self.demo_index])

    def _visualize(self):
        cv2.namedWindow('Visualization', cv2.WINDOW_NORMAL)
        while self.running:
            with self.lock:
                if self.current_image is not None:
                    cv2.imshow('Visualization', cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
                        self.running = False
            cv2.waitKey(25)  # Wait a bit before trying to get the next frame

        cv2.destroyAllWindows()

    def _draw_ee_hand(self, image, robot0_eef, corrected_pose):
        print("=======================================ee=============================================")
        translation = robot0_eef[0:3]
        rotation_quat = robot0_eef[3:7]
        rotation_mat = quat2mat(rotation_quat)
        pose_3 = np.eye(4)
        pose_3[:3, :3] = rotation_mat
        pose_3[:3, 3] = translation
        corrected_pose = corrected_pose.reshape((4, 4))

        hand = robot0_eef[7:]
        #hand = hand.reshape((16, 3))
        hand = apply_pose_matrix(hand, pose_3)

        o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280, 720,
            898.2010498046875,
            897.86669921875,
            657.4981079101562,
            364.30950927734375)

        right_hand_joint_points_homogeneous = np.hstack((hand, np.ones((hand.shape[0], 1))))
        right_hand_transformed_points_homogeneous = np.dot(right_hand_joint_points_homogeneous, np.linalg.inv(corrected_pose).T)
        right_hand_points_to_project = right_hand_transformed_points_homogeneous[:, :3] / right_hand_transformed_points_homogeneous[:, [3]]
        right_hand_back_projected_points = [_back_project_point(point, o3d_depth_intrinsic.intrinsic_matrix) for point in right_hand_points_to_project]

        for i in range(len(right_hand_back_projected_points)):
            u, v = right_hand_back_projected_points[i]
            u = int(float(u) / 1280 * 84)
            v = int(float(v) / 720 * 84)

            if i in [0, 1, 5, 9, 13, 17]:
                cv2.circle(image, (u, v), 2, (0, 255, 0), -1)
            elif i in [4, 8, 12, 16, 20]:
                cv2.circle(image, (u, v), 2, (255, 0, 0), -1)
            else:
                cv2.circle(image, (u, v), 2, (0, 0, 255), -1)
            if i in [1, 2, 3, 4]:
                cv2.circle(image, (u, v), 2, (255, 0, 255), -1)

        return image


    def _draw_action(self, image, robot0_eef, corrected_pose, action, gt=True):
        if gt:
            arrow_color = (0, 0, 255)  # 蓝色
        else:
            arrow_color = (0, 255, 255)  # 黄色

        translation = robot0_eef[0:3]
        rotation_quat = robot0_eef[3:7]
        rotation_mat = quat2mat(rotation_quat)
        pose_3 = np.eye(4)
        pose_3[:3, :3] = rotation_mat
        pose_3[:3, 3] = translation
        corrected_pose = corrected_pose.reshape((4, 4))

        hand_root = np.array([[0.0, 0.0, 0.0]])
        hand_root = apply_pose_matrix(hand_root, pose_3)

        action_trans = action[:3] / 10.0 * 6.0
        hand_root_next = hand_root + action_trans

        hand_root = np.concatenate((hand_root, hand_root_next), axis=0)

        o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280, 720,
            898.2010498046875,
            897.86669921875,
            657.4981079101562,
            364.30950927734375)

        hand_root_points_homogeneous = np.hstack((hand_root, np.ones((hand_root.shape[0], 1))))
        hand_root_homogeneous_transformed_points_homogeneous = np.dot(hand_root_points_homogeneous, np.linalg.inv(corrected_pose).T)
        hand_root_points_to_project = hand_root_homogeneous_transformed_points_homogeneous[:, :3] / hand_root_homogeneous_transformed_points_homogeneous[:, [3]]
        hand_root_back_projected_points = [_back_project_point(point, o3d_depth_intrinsic.intrinsic_matrix) for point in hand_root_points_to_project]

        # 修改箭头的厚度和尖端大小
        arrow_thickness = 4  # 增加箭头的线条宽度
        arrow_tip_length = 0.2  # 增加箭头尖端的大小

        # 绘制箭头
        u0, v0 = hand_root_back_projected_points[0]
        u1, v1 = hand_root_back_projected_points[1]
        cv2.arrowedLine(
            image, 
            [int(float(u0) / 1280 * 84), int(float(v0) / 720 * 84)],  # 起点
            [int(float(u1) / 1280 * 84), int(float(v1) / 720 * 84)],  # 终点
            arrow_color, 
            arrow_thickness,  # 使用更大的线条宽度
            tipLength=arrow_tip_length  # 使用更大的箭头尖端
        )

        # 绘制起点和终点
        for i in range(len(hand_root_back_projected_points)):
            u, v = hand_root_back_projected_points[i]
            u = int(float(u) / 1280 * 84)


            v = int(float(v) / 720 * 84)
            if i in [0]:
                cv2.circle(image, (u, v), 2, (0, 255, 0), -1)  # 绿色起点
            elif i in [1]:
                cv2.circle(image, (u, v), 2, (255, 0, 0), -1)  # 红色终点

        return image

    def start_visualization(self):
        self.running = True
        self.visualization_thread = Thread(target=self._visualize)
        self.visualization_thread.start()

    def stop_visualization(self):
        self.running = False
        self.visualization_thread.join()


def run_trained_agent(args):
    # create environment
    env = VisOnlyEnv(args.dataset_path, obs_horizon=3)
    env.start_visualization()
    env.show_pos()



#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default = '/media/wsw/SSD1T1/ubuntu/model_epoch_3000.pth',
        help="path to saved checkpoint pth file",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/media/wsw/SSD1T1/data/111hand_wiping_1-14_5actiongap_10000points.hdf5',
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
    "--error_path",
    type=str,
    default=None,
    help="(optional) path to save the error log if an error occurs"
)


    args = parser.parse_args()
    res_str = None
    try:
        run_trained_agent(args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        if args.error_path is not None:
            # write traceback to file
            f = open(args.error_path, "w")
            f.write(res_str)
            f.close()
        raise e
    




