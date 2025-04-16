# 本仓库用于HIT-DEX交流使用

# 修改日志
## 2025.04.15
新增对单手采集数据的支持，代码运行顺序如下：
```bash
python redis_glove_server_sing.py
```
```bash
python data_recording_sing.py -v -s --store_hand -o ./save_data_scenario_1 
```
## 2025.04.11
1. 新增了对unitree g1机器人采集的数据（lerobot环境下）进行处理、生成hdf5文件的代码：
   ```bash
    python STEP2_build_dataset/demo_create_hdf5_g1.py
    ```
    注意修改文件夹路径，详细代码参考dataset_utils_g1.py
    注意此代码暂时只支持lerobot数据格式。

2. 新增基于g1的训练代码diffusion_policy_g1，暂时只是在dexcap的dp上进行小范围的修改，可以通过设置config执行此policy的训练：
   ```bash
   python STEP3_train_policy/robomimic/scripts/train.py --config STEP3_train_policy/robomimic/training_config/diffusion_policy_pcd_g1.json
   ```
   注意，后续编写新的algo时记得更新algo/__init__.py、config/__init__.py与相应的config文件。

## 2025.03.30
新增了mujoco仿真功能，
 
运行
```bash
cd STEP3_train_policy/robomimic/scripts
python run_trained_leap_mujoco.py
```
即可在mujoco中查看手部的动作。

运行
```bash
cd STEP3_train_policy/robomimic/scripts
python run_trained_agent_in_mujoco.py
```
即可在用mujoco中搭建的仿真中运行策略。

## 2025.03.19
1. 针对数据处理部分（STEP2），本次对robot0_ee_pos进行了修改，之前的数据是经过robot_to_hand变换过的姿态，现在更新为LEAPHand原点的姿态。这样可以更好的在仿真环境中进行处理，且泛用性增加。（robot_to_hand这个变换我没太看懂，其中包含了一些没有注释的参数，可能是原作者真实环境机械臂末端到手的变换矩阵）
2. 针对训练/验证部分，本次修改了run_trained_agent_vis_withmodel.py，使其能正常运行。并且对训练代码增加了注释。

## 2024.10.14
修改了数据对齐与生成变换矩阵时产生的路径依赖问题，修改了代码执行顺序，使代码能正确运行。

## 2024.10.11
修改了采集手套数据中的字典命名问题，使其能正常与电脑进行通信，并修改传输的关节名称

修改了相机采集时出现的序列号与调动问题，寻找合适的realsense固件以同时启动L515、T265。


# 使用流程：
## 配置硬件
### 手套
配对手套之后，需要修改rokoko studio中的IP地址，同时需要启动redis-server。默认端口是6379。

手套传输的关于手的数据参见data.txt.
根据rokoko官方的回复，我们大拇指可以取4个关节，而其他四根手指可以忽略Metacarpal的数据。

在配置手套时，若出现两只手套的硬件名称一样导致的异常抖动，可以下载Rokoko Legacy在其中修改硬件名称。

手套传输数据选择JOSN v4。
## 相机
相机的配置需要注意，左手相机为相机2，右手相机为相机3，相机序号设置见data_recording.py
## 数据收集
配置好手套、realsense v2.35.2后，在命令行中执行：
```bash
conda activate mocap
cd DexCap/STEP1_collect_data
python redis_glove_server1.py
```
在另一个窗口中启动mocap环境，并执行
```bash
python data_recording.py -s --store_hand -o ./save_data_scenario_1
```
此代码的作用是收集每一帧的的手套数据与相机数据，并将其储存在一起
## 数据处理
### 相机记录数据
初次使用，执行
```bash
cd DexCap/STEP1_collect_data
python replay_human_traj_vis.py --directory save_data_scenario_1 --calib
python calculate_offset_vis_calib.py --directory save_data_scenario_1
```
第一行代码用于将手套数据与图片中的手套进行对齐、校准；第二行代码用于生成校准后的变换矩阵

之后再执行replay_human_traj_vis用于可视化采集到的数据。
```bash
python replay_human_traj_vis.py --directory save_data_scenario_1
```
最后运行来标记任务开始和结束的时间点。
```bash
python demo_clipping_3d.py --directory save_data_scenario_1
```
### 转移到机器人视角
这一步主要用于处理非实验室数据，在Step2数据集构建的dataset_utils.py第155行可以看出，如果在实验室专门的场景采集数
据就不需要再进行此步。

此步的目的是为了防止SLAM建图时采集到桌面一下的数据（这也解释了在实验室中采集数据不需要进行此步）。进行此步骤时需要将桌面数据放到./step1_collect_data/robot_table_pointcloud中，数据结构与采集到的帧一致。
代码运行如下：
```bash
python transform_to_robot_table.py --directory save_data_scenario_1
```
# 注意事项
1. 在采集数据时需要注意，由于手套位置与T265相机绑定，而当速度过快时T265相机会发生偏移，故在采集数据时需要尽量放慢速度，且尽量不翻转手套。

