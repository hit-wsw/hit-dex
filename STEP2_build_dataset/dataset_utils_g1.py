import h5py
import json
from scipy.linalg import svd
from utils import *
from hyperparameters import *


def process_hdf5(output_hdf5_file, metadata_root, action_gap, num_points_to_sample):

    episode_dirs = sorted([
        d for d in os.listdir(metadata_root)
        if d.startswith('episode_')
    ], key=lambda x: int(x.split('_')[1]))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_vis = o3d.geometry.PointCloud()  # Empty point cloud for starters
    firstfirst = True

    with h5py.File(output_hdf5_file, 'w') as output_hdf5:
        output_data_group = output_hdf5.create_group('data')

        demo_index = 0
        total_frames = 0
        mean_init_waist = []
        mean_init_arm = []
        mean_init_hand = []


        for episode in episode_dirs:
            json_path = os.path.join(metadata_root, episode, 'data.json')
            # Load clip marks
            with open(json_path, 'r') as f:
                data = json.load(f)

                agentview_images = []
                pointcloud = []
                labels = []

                waist =[]
                left_arm = []
                right_arm = []
                left_hand = []
                right_hand = []
                states =[]


                for item in data['data']:

                    color_jpg_path = os.path.join(metadata_root, episode, item['colors']['color_0'])
                    depth_png_path = os.path.join(metadata_root, episode, item['colors']['depth_0'])

                    # load pose data
                    waist_data = item['states']['waist']['qpos']
                    left_arm_data = item['states']['left_arm']['qpos']
                    right_arm_data = item['states']['right_arm']['qpos']
                    left_hand_data = item['states']['left_hand']['qpos']
                    right_hand_data = item['states']['right_hand']['qpos']

                    states.append(head_pose.flatten())

                    waist.append(waist_data)
                    left_arm.append(left_arm_data)
                    right_arm.append(right_arm_data)
                    left_hand.append(left_hand_data)
                    right_hand.append(right_hand_data)

                    # process image
                    resized_image = resize_image(color_jpg_path)
                    agentview_images.append(resized_image)

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

                    '''# remove the redundant points bellow the table surface and background
                    centroid = np.mean(robot_table_corner_points, axis=0)
                    A = robot_table_corner_points - centroid
                    U, S, Vt = svd(A)
                    normal = Vt[-1]
                    d = -np.dot(normal, centroid)
                    xyz = color_pcd[:, :3]
                    for plane_gap in table_sweep_list: # sweep over the plane height
                        below_plane = np.dot(xyz, normal[:3]) + d + plane_gap < 0
                        if len(color_pcd[~below_plane]) > num_points_to_sample:
                            color_pcd = color_pcd[~below_plane]
                            break'''

                    # down sample input pointcloud
                    if len(color_pcd) > num_points_to_sample:
                        indices = np.random.choice(len(color_pcd), num_points_to_sample, replace=False)
                        color_pcd = color_pcd[indices]

                    pointcloud.append(copy.deepcopy(color_pcd))
                    labels.append(0)

                    # update pointcloud visualization
                    pcd_vis.points = o3d.utility.Vector3dVector(color_pcd[:, :3])
                    pcd_vis.colors = o3d.utility.Vector3dVector(color_pcd[:, 3:])

                    if firstfirst:
                        vis.add_geometry(pcd_vis)
                        firstfirst = False
                    else:
                        vis.update_geometry(pcd_vis)
                    vis.poll_events()
                    vis.update_renderer()

                    # update image visualization
                    cv2.imshow("resized_image", resized_image)
                    cv2.waitKey(1)
                    length = item['idx'] +1


                
                left_arm = np.array(left_arm)
                right_arm = np.array(right_arm)
                left_hand = np.array(left_hand)
                right_hand = np.array(right_hand)

                waist = np.array(waist)
                robot_arm = np.concatenate((left_arm, right_arm), axis=-1)
                robot_hand = np.concatenate((left_hand, right_hand), axis=-1)


                actions_waist = np.concatenate((waist[action_gap:], waist[-1:].repeat(action_gap, axis=0)), axis=0)
                actions_robot_arm = np.concatenate((robot_arm[action_gap:], robot_arm[-1:].repeat(action_gap, axis=0)), axis=0)
                actions_robot_hand = np.concatenate((robot_hand[action_gap:], robot_hand[-1:].repeat(action_gap, axis=0)), axis=0)

                

                actions = np.concatenate((actions_waist, actions_robot_arm, actions_robot_hand), axis=-1) # merge arm and hand actions

                for j in range(action_gap): # Based on the action_gap, generate the trajectories
                    demo_name = f'demo_{demo_index}'
                    output_demo_group = output_data_group.create_group(demo_name)
                    print("{} saved".format(demo_name))
                    demo_index += 1


                    output_obs_group = output_demo_group.create_group('obs')
                    output_obs_group.create_dataset('agentview_image', data=np.array(agentview_images)[j::action_gap])
                    output_obs_group.create_dataset('pointcloud', data=np.array(pointcloud)[j::action_gap])
                    output_obs_group.create_dataset('waist', data=copy.deepcopy(waist)[j::action_gap])
                    output_obs_group.create_dataset('robot_arm', data=copy.deepcopy(robot_arm)[j::action_gap])
                    output_obs_group.create_dataset('robot_hand', data=copy.deepcopy(robot_hand)[j::action_gap])

                    output_obs_group.create_dataset('label', data=np.array(labels)[j::action_gap])
                    output_demo_group.create_dataset('actions', data=copy.deepcopy(actions)[j::action_gap])

                    # Create 'dones', 'rewards', and 'states'
                    dones = np.zeros(length, dtype=np.int64)
                    dones[-1] = 1  # Set last frame's 'done' to 1
                    output_demo_group.create_dataset('dones', data=dones[j::action_gap])

                    rewards = np.zeros(length, dtype=np.float64)
                    output_demo_group.create_dataset('rewards', data=rewards[j::action_gap])
                    output_demo_group.create_dataset('states', data=states[j::action_gap])

                    output_demo_group.attrs['num_samples'] = len(actions[j::action_gap])

                    total_frames += len(actions[j::action_gap])

                    mean_init_waist.append(copy.deepcopy(waist[j]))
                    mean_init_arm.append(copy.deepcopy(robot_arm[j]))
                    mean_init_hand.append(copy.deepcopy(robot_hand[j]))

        output_data_group.attrs['total'] = total_frames

        # calculate the mean of the initial starting position
        mean_init_waist = np.array(mean_init_waist).mean(axis=0)
        mean_init_arm = np.array(mean_init_arm).mean(axis=0)
        mean_init_hand = np.array(mean_init_hand).mean(axis=0)
        output_data_group.attrs['mean_init_waist'] = mean_init_waist
        output_data_group.attrs['mean_init_arm'] = mean_init_arm
        output_data_group.attrs['mean_init_hand'] = mean_init_hand
