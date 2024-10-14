# 本仓库用于HIT-DEX交流使用

# 使用流程：
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
初次使用，执行
```bash
cd DexCap/STEP1_collect_data
python replay_human_traj_vis.py --directory save_data_scenario_1 --calib
python calculate_offset_vis_calib.py --directory save_data_scenario_1
```
第一行代码用于将手套数据与图片中的手套进行对齐、校准；第二行代码用于生成校准后的变换矩阵

之后再执行
```bash
python replay_human_traj_vis.py --directory save_data_scenario_1
```
用于可视化采集到的数据
# 修改日志
## 2024.10.11
修改了采集手套数据中的字典命名问题，使其能正常与电脑进行通信，并修改传输的关节名称

修改了相机采集时出现的序列号与调动问题，寻找合适的realsense固件以同时启动L515、L256。
## 2024.10.14
修改了数据对齐与生成变换矩阵时产生的路径依赖问题，修改了代码执行顺序，使代码能正确运行。