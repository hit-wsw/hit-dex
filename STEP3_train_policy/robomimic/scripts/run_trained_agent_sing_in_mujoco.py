import sys,mujoco,time,os,json
import numpy as np
import matplotlib.pyplot as plt
from mujoco_parser import *
from utility_mu import *
from transformation import *
from transforms3d.quaternions import mat2quat
import glfw
from transforms3d.euler import quat2mat
from threading import Thread, Lock
from run_trained_agent_utils import _back_project_point, apply_pose_matrix
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS
import torch

np.set_printoptions(precision=2,suppress=True,linewidth=100)
plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)

#xml_path = './asset/realistic_tabletop/scene_table.xml'
xml_path = './STEP3_train_policy/robomimic/scripts/asset/dex_single/scene_dex_sing.xml'
env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path,verbose=True)
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
ckpt_path1 = '/media/wsw/SSD1T1/data/model_epoch_700.pth'

leap_link_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
UR_link_name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

UR_link_init = [-2.39984636,-0.74104342,-2.00760769,-1.96373603,1.57079646,0.74173985]
leap_pos_init =  [-1.80971706, -1.71110369, -1.21016724, -1.03737343, -1.72656429, -1.72750482,
  -1.33077527, -0.98739991, -1.70945451, -1.72752065, -1.46149559, -0.74733658,
  -1.18047625, -1.28725397, -1.28066929, -1.32992703]
leap_pos_init = (np.array(leap_pos_init) + np.pi/2).tolist()

T_265_cam = np.eye(4)
T_265_cam[:3, :3] = np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]])
T_265_cam[:3, 3] = np.array([0.0, 0.076, 0.0])

#-----------------------------cal 265(base) pose
R_cam = rpy2r(np.deg2rad([30,-180.,-90]))
p_cam = np.array([0.9,0.0,1.1])

p_ego  = p_cam ; p_trgt = p_cam + R_cam[:,2] # z-axis forward
T_cam = np.eye(4) ; T_cam[:3,:3] = R_cam; T_cam[:3,3] = p_cam
T_265 = T_cam @ T_265_cam#265 in world

def trans_hand_idx(arr):
    arr[0], arr[1] = arr[1], arr[0]
    arr[4], arr[5] = arr[5], arr[4]
    arr[8], arr[9] = arr[9], arr[8]
    return arr

def trans_pq_2_T(pos,quat):
    T = np.eye(4)
    T[:3,:3] = np.array(quat2mat(quat))
    T[:3,3] = np.array(pos)
    return T

def set_bodys(env,names,joints):
    idxs_step = env.get_idxs_step(joint_names=names)
    env.set_qpos_joints(joint_names=names,qpos=joints)
    env.step(ctrl=joints,ctrl_idxs=idxs_step)

def get_pcd(env):
    ego_rgb_img,ego_depth_img,pcd,_,_ = env.get_egocentric_rgbd_pcd(
        p_ego            = p_ego,
        p_trgt           = p_trgt,
        rsz_rate_for_pcd = 1, # 1/4
        rsz_rate_for_img = 1,
        fovy             = 45, # env.model.cam_fovy[0]
        restore_view     = True,
    )
    #----------------pcd process(mask and transform )--------------------
    pcd = pcd[pcd[:, 2] >= 0.5]
    if pcd.size == 0:
        pcd = np.zeros((20000, 6))
    if len(pcd) > 20000:
        indices = np.random.choice(len(pcd), 20000, replace=False)
        pcd = pcd[indices]
    pcd[:, :3] = np.dot(T_265[:3, :3].T, (pcd[:, :3] - T_265[:3,3]).T).T 
    return pcd,ego_rgb_img,ego_depth_img

def show_pcd(color_pcd):

        # 创建 3D 图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])

        # 提取坐标和颜色
        points = color_pcd[:, :3]
        colors = color_pcd[:, 3:]

        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=1)
        ax.scatter(0, 0, 0)


        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置图形标题
        ax.set_title(f'3D Point Cloud Visualization ')

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
        #time.sleep(1)

        plt.pause(0.5)  # 暂停0.1秒，以便观察每一帧的变化
        plt.close()  # 关闭当前图形，准备绘制下一帧

def get_body_PQ(env,name):
    p_ee,R_ee = env.get_pR_body(body_name=name)
    T_ee = np.eye(4);T_ee[:3,:3] = R_ee; T_ee[:3,3] = p_ee #ee pose in world
    T_ee_265 = np.linalg.inv(T_265) @ T_ee
    
    ee_pos = T_ee_265[:3,3]; ee_quat = mat2quat(T_ee_265[:3,:3])
    return ee_pos,ee_quat

def IK(env,joint_names,body_name_trgt,q_ik_init,p,R):

    qpos,ik_err_stack,ik_info = solve_ik(
        env                = env,
        joint_names_for_ik = joint_names,
        body_name_trgt     = body_name_trgt,
        q_init             = q_ik_init,
        p_trgt             = p,
        R_trgt             = R,
        max_ik_tick        = 500,
        ik_stepsize        = 1.0,
        ik_eps             = 1e-2,
        ik_th              = np.radians(5.0),
        render             = False,
        verbose_warning    = False,
    )
    return qpos


#------------Move UR to the right place------------------------
env.reset()
env.set_p_body(body_name='ur_base',p=np.array([0,0,0.5])) # move UR

#-----------------------------load Objects----------------------------------
env.set_p_base_body(body_name='obj_box',p=[0.4,0.3,0.55])
env.set_R_base_body(body_name='obj_box',R=np.eye(3,3))

env.set_p_base_body(body_name='obj_redcube',p=[0.4,-0.3,0.51])
env.set_R_base_body(body_name='obj_redcube',R=np.eye(3,3))
#-----------------------control UR to init place-------------------------------

set_bodys(env,UR_link_name,UR_link_init)
set_bodys(env,leap_link_name,leap_pos_init)

#---------------------load sliders to ctrl UR----------------------------
idxs_fwd = env.get_idxs_fwd(joint_names=env.rev_joint_names)
#--------------------init mujoco sim env--------------------
env.init_viewer(
    transparent = False,
    azimuth     = 105,
    distance    = 3.12,
    elevation   = -29,
    lookat      = [0.39, 0.25, 0.43],
)

collect_data = False
step_counter = 0
pointcloud = [];robot0_eef_pos = [];robot0_eef_quat = [];robot0_eef_hand = []

# 定义键盘回调函数
def key_callback(window, key, scancode, action, mods):
    global collect_data, step_counter
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        collect_data = True
        step_counter = 0
        print("Started collecting 3-step data...")

# 设置键盘回调
if hasattr(env.viewer, "window"):
    glfw.set_key_callback(env.viewer.window, key_callback)
else:
    print("Warning: Viewer does not support GLFW key callbacks.")

algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=ckpt_path1)
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
policy.start_episode()
# 在渲染循环中添加键盘检测
# 主循环
while env.is_viewer_alive():
    init_qpos = env.get_qpos_joints(joint_names=env.rev_joint_names)
    env.forward(q=init_qpos, joint_idxs=idxs_fwd)

    
    if collect_data:
        pcd, _, _ = get_pcd(env)
        show_pcd(pcd)
        ee_pos, ee_quat = get_body_PQ(env, 'leap_base')
        hand_pos = env.get_qpos_joints(joint_names=leap_link_name)
        hand_pos = (np.array(hand_pos) + np.pi/2).tolist()
        
        pointcloud.append(pcd)
        robot0_eef_pos.append(ee_pos)
        robot0_eef_quat.append(ee_quat)
        robot0_eef_hand.append(hand_pos)
        
        if len(pointcloud) > 3:
            pointcloud.pop(0)
            robot0_eef_pos.pop(0)
            robot0_eef_quat.pop(0)
            robot0_eef_hand.pop(0)
            
        if len(pointcloud) == 3:
            print("Data collection completed!")
            pointcloud1 = np.array(pointcloud)
            robot0_eef_pos1 = np.array(robot0_eef_pos)
            robot0_eef_quat1 = np.array(robot0_eef_quat)
            robot0_eef_hand1 = np.array(robot0_eef_hand)

            pointcloud1 = TensorUtils.to_device(torch.FloatTensor(pointcloud1),device)
            robot0_eef_pos1 = TensorUtils.to_device(torch.FloatTensor(robot0_eef_pos1),device)
            robot0_eef_quat1 = TensorUtils.to_device(torch.FloatTensor(robot0_eef_quat1),device)
            robot0_eef_hand1 = TensorUtils.to_device(torch.FloatTensor(robot0_eef_hand1),device)


            state = {
                'pointcloud': pointcloud1,
                'robot0_eef_pos': robot0_eef_pos1,
                'robot0_eef_quat': robot0_eef_quat1,
                'robot0_eef_hand': robot0_eef_hand1,
            }
            
            # 可以在这里保存 return_state 或进行其他处理
            action = policy(ob = state)
            pose_right = action[0:3]
            quat_right = action[3:7]
            hand_right = action[7:]
            hand_right = (np.array(hand_right) + np.pi/2).tolist()
            hand_right = trans_hand_idx(hand_right)
            T_ee_r = trans_pq_2_T(pose_right,quat_right)
            T_ee_r_world = T_265 @ T_ee_r
            qpos_r = IK(env,UR_link_name,'leap_base',init_qpos[:6],T_ee_r_world[:3,3],T_ee_r_world[:3,:3])
            init_qpos[6:]=hand_right
            env.forward(q=init_qpos, joint_idxs=idxs_fwd)
        
    
    env.plot_T()
    env.plot_T(p=T_265[:3,3], R=T_265[:3,:3], axis_len=0.1, label='T265')
    env.plot_T(p=T_cam[:3,3], R=T_cam[:3,:3], axis_len=0.1, label='L515')
    env.render()
    collect_data = True
