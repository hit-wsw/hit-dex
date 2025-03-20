# 本仓库用于HIT-DEX交流使用

# 使用流程：
## 配置硬件
### 手套
配对手套之后，需要修改rokoko studio中的IP地址，同时需要启动redis-server。默认端口是6379。

手套传输的关于手的数据如下：
```python
{"version":"3,0","fps":30.0,"scene":
 {"timestamp":29.3385086,"actors":[],"newtons":[{
    "name":"man","color":[248,248,248],
    "meta":{"hasHands":true,"hasLeftHand":true,"hasRightHand":false,"hasBody":false,"hasFace":false},
    "dimensions":{"totalHeight":1.8,"hipHeight":0.9559322},
    "joints":[                                                                                                                                                                                                     
    {"name":"LeftHand","parent":40,"position":{"x":-0.401281357,"y":1.30209112,"z":0.243864283},"rotation":{"x":-0.09616978,"y":-0.676203132,"z":0.614187837,"w":-0.39531514}},
    {"name":"LeftFinger5Metacarpal","parent":41,"position":{"x":-0.4267673,"y":1.28991067,"z":0.25323385},"rotation":{"x":-0.2323878,"y":-0.6358294,"z":0.4526294,"w":-0.5803822}},
    {"name":"LeftFinger5Proximal","parent":42,"position":{"x":-0.467341661,"y":1.26607525,"z":0.268350035},"rotation":{"x":-0.239234269,"y":-0.53119427,"z":0.692901,"w":-0.424838454}},
    {"name":"LeftFinger5Medial","parent":43,"position":{"x":-0.498063415,"y":1.26879728,"z":0.287771434},"rotation":{"x":-0.233213767,"y":-0.571949542,"z":0.6025368,"w":-0.5054055}},
    {"name":"LeftFinger5Distal","parent":44,"position":{"x":-0.518128335,"y":1.26501441,"z":0.2981612},"rotation":{"x":-0.199519545,"y":-0.6102844,"z":0.5636752,"w":-0.51963}},
    {"name":"LeftFinger5Tip","parent":45,"position":{"x":-0.5374818,"y":1.25836527,"z":0.30937764},"rotation":{"x":-0.1995193,"y":-0.610284448,"z":0.563675165,"w":-0.5196301}},
    {"name":"LeftFinger4Metacarpal","parent":41,"position":{"x":-0.4219265,"y":1.29274559,"z":0.261505216},"rotation":{"x":-0.16576831,"y":-0.6618694,"z":0.5401571,"w":-0.492625624}},
    {"name":"LeftFinger4Proximal","parent":47,"position":{"x":-0.461549222,"y":1.2736882,"z":0.2905887},"rotation":{"x":-0.1909398,"y":-0.583674967,"z":0.698358238,"w":-0.367642641}},
    {"name":"LeftFinger4Medial","parent":48,"position":{"x":-0.494718373,"y":1.2758652,"z":0.320985228},"rotation":{"x":-0.174976185,"y":-0.6556831,"z":0.577713,"w":-0.453553438}},
    {"name":"LeftFinger4Distal","parent":49,"position":{"x":-0.5166073,"y":1.267985,"z":0.338382155},"rotation":{"x":-0.130691692,"y":-0.7080443,"z":0.512197137,"w":-0.468238264}},
    {"name":"LeftFinger4Tip","parent":50,"position":{"x":-0.535408,"y":1.25550842,"z":0.355434567},"rotation":{"x":-0.130691841,"y":-0.7080441,"z":0.5121974,"w":-0.4682382}},
    {"name":"LeftFinger3Metacarpal","parent":41,"position":{"x":-0.414583921,"y":1.2966733,"z":0.269010156},"rotation":{"x":-0.09616978,"y":-0.676203132,"z":0.614187837,"w":-0.39531514}},
    {"name":"LeftFinger3Proximal","parent":52,"position":{"x":-0.4484739,"y":1.2841748,"z":0.310547978},"rotation":{"x":-0.165780574,"y":-0.611453652,"z":0.6759205,"w":-0.376527041}},
    {"name":"LeftFinger3Medial","parent":53,"position":{"x":-0.4827966,"y":1.28266585,"z":0.344388723},"rotation":{"x":-0.130900934,"y":-0.6775714,"z":0.6027778,"w":-0.400525928}},
    {"name":"LeftFinger3Distal","parent":54,"position":{"x":-0.5037972,"y":1.27506232,"z":0.3670351},"rotation":{"x":-0.102087818,"y":-0.7187205,"z":0.55306834,"w":-0.4088209}},
    {"name":"LeftFinger3Tip","parent":55,"position":{"x":-0.522223055,"y":1.26376045,"z":0.388923854},"rotation":{"x":-0.102087915,"y":-0.7187204,"z":0.5530685,"w":-0.408820868}},
    {"name":"LeftFinger2Metacarpal","parent":41,"position":{"x":-0.4065476,"y":1.300819,"z":0.275157362},"rotation":{"x":-0.009882731,"y":-0.6863157,"z":0.6624718,"w":-0.300007135}},
    {"name":"LeftFinger2Proximal","parent":57,"position":{"x":-0.4297782,"y":1.29392064,"z":0.326211452},"rotation":{"x":-0.105102226,"y":-0.750014961,"z":0.572588861,"w":-0.313963383}},
    {"name":"LeftFinger2Medial","parent":58,"position":{"x":-0.454073161,"y":1.278786,"z":0.363457382},"rotation":{"x":-0.09569154,"y":-0.7921345,"z":0.499479115,"w":-0.337471128}},
    {"name":"LeftFinger2Distal","parent":59,"position":{"x":-0.468197078,"y":1.26483524,"z":0.3844593},"rotation":{"x":-0.07653578,"y":-0.8190046,"z":0.454079062,"w":-0.342324317}},
    {"name":"LeftFinger2Tip","parent":60,"position":{"x":-0.480552971,"y":1.24852479,"z":0.404040456},"rotation":{"x":-0.07653489,"y":-0.8190039,"z":0.454079449,"w":-0.342325568}},
    {"name":"LeftFinger1Metacarpal","parent":41,"position":{"x":-0.394852966,"y":1.306451,"z":0.278675675},"rotation":{"x":-0.08430527,"y":-0.8359901,"z":0.478677452,"w":0.254717171}},
    {"name":"LeftFinger1Proximal","parent":62,"position":{"x":-0.390672117,"y":1.28501749,"z":0.312938273},"rotation":{"x":0.247972265,"y":0.801229954,"z":-0.5219023,"w":-0.155428559}},
    {"name":"LeftFinger1Distal","parent":63,"position":{"x":-0.397923023,"y":1.274771,"z":0.341106832},"rotation":{"x":0.247684821,"y":0.800264835,"z":-0.5233812,"w":-0.155886263}},
    {"name":"LeftFinger1Tip","parent":64,"position":{"x":-0.406140238,"y":1.2631644,"z":0.373339534},"rotation":{"x":0.247684151,"y":0.8002632,"z":-0.523383558,"w":-0.15588747}},                                                                                                                                                       
    ]}],"props":[],"characters":[]}}
```
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

之后再执行
```bash
python replay_human_traj_vis.py --directory save_data_scenario_1
```
用于可视化采集到的数据。
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
# 修改日志
## 2024.10.11
修改了采集手套数据中的字典命名问题，使其能正常与电脑进行通信，并修改传输的关节名称

修改了相机采集时出现的序列号与调动问题，寻找合适的realsense固件以同时启动L515、T265。
## 2024.10.14
修改了数据对齐与生成变换矩阵时产生的路径依赖问题，修改了代码执行顺序，使代码能正确运行。
## 2025.03.19
咕咕几个月又更新了！本次更新内容如下：
1. 针对数据处理部分（STEP2），本次对robot0_ee_pos进行了修改，之前的数据是经过robot_to_hand变换过的姿态，现在更新为LEAPHand原点的姿态。这样可以更好的在仿真环境中进行处理，且泛用性增加。（robot_to_hand这个变换我没太看懂，其中包含了一些没有注释的参数，可能是原作者真实环境机械臂末端到手的变换矩阵）
2. 针对训练/验证部分，本次修改了run_trained_agent_vis_withmodel.py，使其能正常运行。并且对训练代码增加了注释。