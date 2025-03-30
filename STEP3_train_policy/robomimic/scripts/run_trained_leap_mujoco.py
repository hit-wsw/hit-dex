# ffmpeg -framerate 10 -i saved_current_frames/%d.jpg -c:v mpeg4 -pix_fmt yuv420p saved_run_video.mp4

import argparse
import h5py
import time
import traceback
import numpy as np
import cv2
import torch
import open3d as o3d
from transforms3d.euler import quat2mat
from threading import Thread, Lock
from run_trained_agent_utils import _back_project_point, apply_pose_matrix
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS
from mujoco_parser import *
from utility_mu import *
from transformation import *


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

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                # 获取最后一个点云数据
                last_pcd = pointcloud[-1]
                
                # 绘制点云
                ax.scatter(last_pcd[:, 0], last_pcd[:, 1], last_pcd[:, 2], 
                        c=last_pcd[:, 3:6], s=1, marker='.')
                
                ax.scatter([0], [0], [0], c='red', s=100, marker='o', label='Origin (0,0,0)')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Last Collected Point Cloud')
                
                
                # 调整视角
                ax.view_init(elev=20, azim=45)
                
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)

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

    xml_path = './STEP3_train_policy/robomimic/scripts/asset/leap_hand_mesh/scene_floor.xml'
    env_mu = MuJoCoParserClass(name='LEAP',rel_xml_path=xml_path,verbose=True)
    leap_link_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

    env_mu.set_p_body(body_name='leap_base',p=np.array([0,0,0.5]))
    idxs_fwd = env_mu.get_idxs_fwd(joint_names=leap_link_name)
    env_mu.init_viewer(transparent=True)

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

    while True:
        if env.demo_index > 5:
            break
        state, done = env.get_state()

        cur = time.time()
        action = policy(ob = state)
        times = time.time()-cur
        print('time:[%f]'%times)

        #print("++++++++++++++++++++++++POSE RIGHT+++++++++++++++++++++++")
        pose_right = action[0:3]
        #print(action[0:3])
        #print("++++++++++++++++++++++++POSE LEFT+++++++++++++++++++++++")
        pose_left = action[3:6]
        #print(action[3:6])
        #print("++++++++++++++++++++++++QUAT RIGHT+++++++++++++++++++++++")
        quat_right = action[6:10]
        #print(action[6:10])
        #print("++++++++++++++++++++++++QUAT LEFT+++++++++++++++++++++++")
        quat_left = action[10:14]
        #print(action[10:14])
        #print("++++++++++++++++++++++++HAND RIGHT+++++++++++++++++++++++")
        hand_right = action[14:30]
        #print(action[14:30])
        #print("++++++++++++++++++++++++HAND LEFT+++++++++++++++++++++++")
        hand_left = action[30:]
        #print(action[30:])
        '''print("++++++++++++++++++++++++HAND RIGHT+++++++++++++++++++++++")
        print((state['robot0_eef_hand'][-1][:16].cpu().numpy() - hand_right).max())

        print("++++++++++++++++++++++++HAND RIGHT+++++++++++++++++++++++")
        print((state['robot0_eef_pos'][-1][:3].cpu().numpy() - pose_right).max())

        print("++++++++++++++++++++++++HAND RIGHT+++++++++++++++++++++++")
        print((state['robot0_eef_quat'][-1][:4].cpu().numpy() - quat_right).max())

        print("++++++++++++++++++++++++HAND LEFT+++++++++++++++++++++++")
        print((state['robot0_eef_hand'][-1][16:].cpu().numpy() - hand_left).max())

        print("++++++++++++++++++++++++HAND LEFT+++++++++++++++++++++++")
        print((state['robot0_eef_pos'][-1][3:].cpu().numpy() - pose_left).max())

        print("++++++++++++++++++++++++HAND LEFT+++++++++++++++++++++++")
        print((state['robot0_eef_quat'][-1][4:].cpu().numpy() - quat_left).max())
        print("\n")
        print("\n")
        print("\n")'''

        #env_mu.set_R_base_body(body_name='leap_base',R=quat2mat(quat_right))
        #env_mu.set_p_body(body_name='leap_base',p=pose_right)
        env_mu.forward(q=hand_right,joint_idxs=idxs_fwd)
        
        if env_mu.loop_every(tick_every=1):
            env_mu.plot_T()
            env_mu.plot_body_T(body_name='leap_base')
            env_mu.plot_joint_axis(axis_len=0.025,axis_r=0.005) # revolute joints
            env_mu.render()
        #env.save_current_frame()

        if done:
            env.reset()
            policy.start_episode()

        time.sleep(0.1)


#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default = '/media/wsw/SSD1T1/data/1-20_model/model_epoch_3000.pth',
        help="path to saved checkpoint pth file",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/media/wsw/SSD1T1/data/hand_packaging_wild_1-20_5ag_10000points.hdf5',
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