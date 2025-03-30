import sys,mujoco,time,os,json
import numpy as np
import matplotlib.pyplot as plt
from mujoco_parser import *
from utility_mu import *
from transformation import *
from transforms3d.quaternions import mat2quat
xml_path = './asset/leap_hand_mesh/scene_floor.xml'
env = MuJoCoParserClass(name='LEAP',rel_xml_path=xml_path,verbose=True)
leap_link_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

env.set_p_body(body_name='leap_base',p=np.array([0,0,0.5]))
idxs_fwd = env.get_idxs_fwd(joint_names=leap_link_name)
# Loop
env.init_viewer(transparent=True)
while env.is_viewer_alive():
    # Update
    qpos = np.random.uniform(low=-1, high=1, size=(1, 16))
    env.forward(q=qpos,joint_idxs=idxs_fwd)
    # Render
    if env.loop_every(tick_every=10):
        env.plot_T()
        env.plot_body_T(body_name='leap_base')
        env.plot_joint_axis(axis_len=0.025,axis_r=0.005) # revolute joints
        env.render()
# Close slider
print ("Done.")
