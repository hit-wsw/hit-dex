# ffmpeg -framerate 10 -i saved_current_frames/%d.jpg -c:v mpeg4 -pix_fmt yuv420p saved_run_video.mp4

import argparse
import h5py
import time
import traceback
import numpy as np
import cv2
import torch
from threading import Thread, Lock
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import matplotlib.pyplot as plt

joints = [
    'posx','pos_y','pos_z',
    'quat_x', 'quat_y', 'quat_z', 'quat_w',
    'LEAP_0', 'LEAP_1', 'LEAP_2', 'LEAP_3', 'LEAP_4', 'LEAP_5',
    'LEAP_6', 'LEAP_7', 'LEAP_8', 'LEAP_9', 'LEAP_10', 'LEAP_11', 
    'LEAP_12', 'LEAP_13', 'LEAP_14', 'LEAP_15'
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
            self.pos_data = demo_group['obs']['robot0_eef_pos'][:]
            self.quat_data = demo_group['obs']['robot0_eef_quat'][:]
            self.hand_data = demo_group['obs']['robot0_eef_hand'][:]
            self.robot0_eef = np.concatenate((self.pos_data, self.quat_data, self.hand_data), axis=-1)

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
                robot_pos = self.pos_data[(self.current_index-self.obs_horizon):self.current_index]
                robot_quat = self.quat_data[(self.current_index-self.obs_horizon):self.current_index]
                robot_hand = self.hand_data[(self.current_index-self.obs_horizon):self.current_index]

                # postprocess data frames
                #agentview_image = ObsUtils.batch_image_hwc_to_chw(agentview_image) / 255.0
                #agentview_image = TensorUtils.to_device(torch.FloatTensor(agentview_image), self.device)

                pointcloud = TensorUtils.to_device(torch.FloatTensor(pointcloud), self.device)
                robot_pos = TensorUtils.to_device(torch.FloatTensor(robot_pos), self.device)
                robot_quat = TensorUtils.to_device(torch.FloatTensor(robot_quat), self.device)
                robot_hand = TensorUtils.to_device(torch.FloatTensor(robot_hand), self.device)

                return_state = {
                    'pointcloud': pointcloud,
                    'robot0_eef_pos': robot_pos,
                    'robot0_eef_quat': robot_quat,
                    'robot0_eef_hand': robot_hand,
                }

                if self.current_index < len(self.agentview_image_data)-1:
                    robot_pos_ac = self.pos_data[self.current_index]
                    robot_quat_ac = self.quat_data[self.current_index]
                    robot_hand_ac = self.hand_data[self.current_index]
                    state_ac = np.concatenate((robot_pos_ac,robot_quat_ac,robot_hand_ac), axis=-1)

                else:
                    robot_pos_ac = self.pos_data[self.current_index-1]
                    robot_quat_ac = self.quat_data[self.current_index-1]
                    robot_hand_ac = self.hand_data[self.current_index-1]
                    state_ac = np.concatenate((robot_pos_ac,robot_quat_ac,robot_hand_ac), axis=-1)

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
                filename = f"/home/wsw/Dexcap (copy)/pic_sing_ball/demo_{env.demo_index}_{joint_name}.png"
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
        default = '/media/wsw/SSD1T1/data/model_epoch_700.pth',
        help="path to saved checkpoint pth file",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/media/wsw/SSD1T1/data/grasp_ball_5actiongap_20000points.hdf5',
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