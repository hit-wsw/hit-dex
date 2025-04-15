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

joints = [
    'Waist',
    'Arm_L1', 'Arm_L2', 'Arm_L3', 'Arm_L4', 'Arm_L5', 'Arm_L6', 'Arm_L7',
    'Arm_R1', 'Arm_R2', 'Arm_R3', 'Arm_R4', 'Arm_R5', 'Arm_R6', 'Arm_R7',
    'Hand_L1', 'Hand_L2', 'Hand_L3', 'Hand_L4', 'Hand_L5', 'Hand_L6',
    'Hand_R1', 'Hand_R2', 'Hand_R3', 'Hand_R4', 'Hand_R5', 'Hand_R6'
]

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
            self.waist_data = demo_group['obs']['waist'][:]
            self.robot_arm_data = demo_group['obs']['robot_arm'][:]
            self.robot_hand_data = demo_group['obs']['robot_hand'][:]
            self.robot0_eef = np.concatenate((self.waist_data, self.robot_arm_data, self.robot_hand_data), axis=-1)

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

                if self.current_index < len(self.agentview_image_data)-1:
                    waist_ac = self.waist_data[self.current_index]
                    robot_arm_ac = self.robot_arm_data[self.current_index]
                    robot_hand_ac = self.robot_hand_data[self.current_index]
                    state_ac = np.concatenate((waist_ac,robot_arm_ac,robot_hand_ac), axis=-1)

                else:
                    waist_ac = self.waist_data[self.current_index-1]
                    robot_arm_ac = self.robot_arm_data[self.current_index-1]
                    robot_hand_ac = self.robot_hand_data[self.current_index-1]
                    state_ac = np.concatenate((waist_ac,robot_arm_ac,robot_hand_ac), axis=-1)

                done = bool(self.done_data[self.current_index])
                self.current_index += 1
                return return_state, state_ac, done
            else:
                return None, None, True  # Return True for done if no more images

    def vis_model_action(self, action):
        self.current_image = self._draw_action(self.current_image, self.robot0_eef[self.current_index - 2], self.state_data[self.current_index - 2], action, gt=False)
    def vis_gt_action(self):
        self.current_image = self._draw_action(self.current_image, self.robot0_eef[self.current_index - 2], self.state_data[self.current_index - 2], self.action_data[self.current_index - 2])

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


    def start_visualization(self):
        self.running = True
        self.visualization_thread = Thread(target=self._visualize)
        self.visualization_thread.start()

    def stop_visualization(self):
        self.running = False
        self.visualization_thread.join()


def run_trained_agent(args):
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    policy.start_episode()

    # create environment
    env = VisOnlyEnv(args.dataset_path, obs_horizon=3)
    env.start_visualization()

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    action_list = []
    state_ac_list = []
    while True:
        if env.demo_index > 5:
            break
        
        state, state_ac, done = env.get_state()
        action = policy(ob=state)
        action_list.append(action)
        state_ac_list.append(state_ac)

        if done:
            # 转换为NumPy数组方便处理
            action_array = np.array(action_list[:-1])  # 形状: (T, 26)
            state_ac_array = np.array(state_ac_list[:-1])  # 形状: (T, 26)
            timesteps = np.arange(len(action_list)-1)  # 时间步序列
            
            # 为每个关节绘制时序对比图
            for joint_idx, joint_name in enumerate(joints):
                plt.figure(figsize=(10, 5))
                
                # 绘制Action和State_ac的时序曲线
                plt.plot(timesteps, action_array[:, joint_idx], 
                        label='Action', color='blue', marker='o')
                plt.plot(timesteps, state_ac_array[:, joint_idx], 
                        label='State_ac', color='orange', marker='x')
                
                # 绘制差异区域（填充）
                plt.fill_between(
                    timesteps,
                    action_array[:, joint_idx],
                    state_ac_array[:, joint_idx],
                    color='red', alpha=0.2, label='Difference (Action - State_ac)'
                )
                
                plt.title(f"Demo {env.demo_index} | Joint: {joint_name}")
                plt.xlabel("Time Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                
                # 保存图片
                filename = f"/home/wsw/Dexcap (copy)/pic/demo_{env.demo_index}_{joint_name}.png"
                plt.savefig(filename, bbox_inches='tight', dpi=100)
                plt.close()
            env.reset()
            policy.start_episode()
            print('demo done')
            action_list = []
            state_ac_list = []

        time.sleep(0.1)


#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default = '/media/wsw/SSD1T1/data/diffusion_policy_pcd_g1/20250412105806/models/model_epoch_3000.pth',
        help="path to saved checkpoint pth file",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/media/wsw/SSD1T1/data/g1_2actiongap_10000points.hdf5',
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