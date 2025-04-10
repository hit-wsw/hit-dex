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
xml_path = './STEP3_train_policy/robomimic/scripts/asset/dex_single/scene_dex_kine.xml'
env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path,verbose=True)
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
ckpt_path1 = '/media/wsw/SSD1T1/data/1-20_model/model_epoch_3000.pth'

UR_link_l = ['shoulder_pan_joint_l','shoulder_lift_joint_l','elbow_joint_l','wrist_1_joint_l','wrist_2_joint_l','wrist_3_joint_l']
leap_link_name_l = ['0_l', '1_l', '2_l', '3_l', '4_l', '5_l', '6_l', '7_l', '8_l', '9_l', '10_l', '11_l', '12_l', '13_l', '14_l', '15_l']

leap_link_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
UR_link_name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

UR_link_init = [-0.30229427 ,-1.70501379  ,1.53648902 ,-1.40229984 ,-1.57079478 ,-0.30228947]
UR_link_init_l = [-2.83201632 ,-1.4262628 , -1.53125663 ,-1.75492036 , 1.57081689 , 0.30953829]
leap_pos_init = [-0,0.01,0,-0,0,0.01,0,0,0,0.01,0,0,0.83,-0.08,-0,0]

T_265_cam = np.eye(4)
T_265_cam[:3, :3] = np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]])
T_265_cam[:3, 3] = np.array([0.0, 0.076, 0.0])

#-----------------------------cal 265(base) pose
R_cam = rpy2r(np.deg2rad([-125.22,-0.,-90]))
p_cam = np.array([0.05,0.0,1.05])

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
    pcd = pcd[pcd[:, 2] >= 0.51]
    if pcd.size == 0:
        pcd = np.zeros((10000, 6))
    if len(pcd) > 10000:
        indices = np.random.choice(len(pcd), 10000, replace=False)
        pcd = pcd[indices]
    pcd[:, :3] = np.dot(T_265[:3, :3].T, (pcd[:, :3] - T_265[:3,3]).T).T 
    return pcd,ego_rgb_img,ego_depth_img

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
env.set_p_body(body_name='ur_base',p=np.array([0,-0.3,0.5])) # move UR
env.set_p_body(body_name='ur_base_l',p=np.array([0,0.3,0.5]))

#-----------------------------load Objects----------------------------------
env.set_p_base_body(body_name='obj_box',p=[0.45,0.25,0.51])
env.set_R_base_body(body_name='obj_box',R=np.eye(3,3))

env.set_p_base_body(body_name='obj_redcube',p=[0.4,-0.1,0.51])
env.set_R_base_body(body_name='obj_redcube',R=np.eye(3,3))
#-----------------------control UR to init place-------------------------------

set_bodys(env,UR_link_name,UR_link_init)
set_bodys(env,leap_link_name,leap_pos_init)
set_bodys(env,UR_link_l,UR_link_init_l)
set_bodys(env,leap_link_name_l,leap_pos_init)

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

    
    if collect_data :
        pcd, _, _ = get_pcd(env)
        ee_pos, ee_quat = get_body_PQ(env, 'leap_base')
        ee_pos_l, ee_quat_l = get_body_PQ(env, 'leap_base_l')
        hand_pos = env.get_qpos_joints(joint_names=leap_link_name)
        hand_pos_l = env.get_qpos_joints(joint_names=leap_link_name_l)
        
        pointcloud.append(pcd)
        robot0_eef_pos.append(np.concatenate([ee_pos, ee_pos_l]))
        robot0_eef_quat.append(np.concatenate([ee_quat, ee_quat_l]))
        robot0_eef_hand.append(np.concatenate([hand_pos, hand_pos_l]))
        
        step_counter += 1
        if step_counter == 3:
            print("Data collection completed!")
            pointcloud = np.array(pointcloud)
            robot0_eef_pos = np.array(robot0_eef_pos)
            robot0_eef_quat = np.array(robot0_eef_quat)
            robot0_eef_hand = np.array(robot0_eef_hand)

            pointcloud = TensorUtils.to_device(torch.FloatTensor(pointcloud),device)
            robot0_eef_pos = TensorUtils.to_device(torch.FloatTensor(robot0_eef_pos),device)
            robot0_eef_quat = TensorUtils.to_device(torch.FloatTensor(robot0_eef_quat),device)
            robot0_eef_hand = TensorUtils.to_device(torch.FloatTensor(robot0_eef_hand),device)


            state = {
                'pointcloud': pointcloud,
                'robot0_eef_pos': robot0_eef_pos,
                'robot0_eef_quat': robot0_eef_quat,
                'robot0_eef_hand': robot0_eef_hand,
            }
            
            # 可以在这里保存 return_state 或进行其他处理
            action = policy(ob = state)
            pose_right = action[0:3];pose_left = action[3:6]
            quat_right = action[6:10];quat_left = action[10:14]
            hand_right = action[14:30];hand_left = action[30:]
            hand_right = trans_hand_idx(hand_right);hand_left = trans_hand_idx(hand_left)
            T_ee_r = trans_pq_2_T(pose_right,quat_right);T_ee_l = trans_pq_2_T(pose_left,quat_left)
            T_ee_r_world = T_265 @ T_ee_r ; T_ee_l_world = T_265 @ T_ee_l 
            qpos_r = IK(env,UR_link_name,'leap_base',init_qpos[:6],T_ee_r_world[:3,3],T_ee_r_world[:3,:3])
            qpos_l = IK(env,UR_link_l,'leap_base_l',init_qpos[22:28],T_ee_l_world[:3,3],T_ee_l_world[:3,:3])
            
            init_qpos[:6] = qpos_r;init_qpos[22:28]=qpos_l;init_qpos[6:22]=hand_right;init_qpos[28:]=hand_left
            env.forward(q=init_qpos, joint_idxs=idxs_fwd)
            pointcloud = [];robot0_eef_pos = [];robot0_eef_quat = [];robot0_eef_hand = [];step_counter=0
            print(pointcloud)
        
    
    env.plot_T()
    env.plot_T(p=T_265[:3,3], R=T_265[:3,:3], axis_len=0.1, label='Real 265')
    env.render()
