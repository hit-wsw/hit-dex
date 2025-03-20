diffusion_policy中的obs_encoder用于处理观察数据，是一个ObservationGroupEncoder类，在models文件夹中的obs_nets.py，其又与ObservationEncoder、ObservationDecoder两个编码器与解码器类有关。下面根据顺序给出每个类的解释。

注意：观察数据的类型很多，比如robot0_eef_pos 观察键与三通道 RGB “agentview_image ”观察键。所以使用了torch Modules模块，


# obs_encoder_factory函数
根据输入的观测形状和配置参数，动态创建并配置一个ObservationEncoder
## 函数接收参数：
1. obs_shapes：OrderedDict类型，将观察数据的名称映射到形状（image->(3,244,244)）
2. feature_activation：激活函数，默认是nn.ReLU
3. encoder_kwargs：自定义的编码器参数，包含每种模态的编码器配置。如果是None，则使用默认配置。
encoder_kwargs的具体结构如下：
```python
obs_modality1: dict
    feature_dimension: int
    core_class: str
    core_kwargs: dict
        ...
        ...
    obs_randomizer_class: str
    obs_randomizer_kwargs: dict
        ...
        ...
obs_modality2: dict
    ...
```
其中obs_modality1和obs_modality2分别表示不同的观察模态。每种模态对应一个字典。每个键值对的解释如下:
1. feature_dimension：指定该模态编码后生成的特征向量的维度（输出维度）
2. core_class：指定核心编码器类的名称，对不同的模态使用不同的编码器（图像：CNN，点云：Pointnet）
3. core_kwargs：字典，核心编码器的初始化参数（层数大小、通道数等）
4. obs_randomizer_class:指定随机扰动器类的名称，用于对输入数据进行随机化处理（噪声or旋转）
5. bs_randomizer_kwargs：字典，随机扰动器类指定初始化参数

## 函数内容
1. 实例化ObservationEncoder（下面会说），记为enc，传入指定的激活函数feature_activation
2. 遍历obs_shapes字典，通过ObsUtils.OBS_KEYS_TO_MODALITIES确定观察的模态类型，并用encoder_kwargs为每个模态拷贝编码器参数为enc_kwargs。
3. 对每个模态，配置编码器核心和扰动器，将其添加到encoder_kwargs中。
4. 对点云模态进行特殊化处理：使用PointNet作为核心编码器，点云输入维度为6，输出维度为64，隐藏层为[32, 64, 128, 256]
5. 调用register_obs_key 方法，将当前观察数据的相关信息（名称、形状、编码器类和参数、随机扰动器）注册到 enc 对象中。
6. enc.make()并返回enc


# ObservationEncoder类
该类用于处理模仿学习中的观测数据（observation data），将处理过的多个观测键的结果拼接在一起
## __init__构造函数
该构造函数接收8个参数：
1. obs_shapes：OrderedDict，储存观测键（名称与形状）
2. obs_nets_classes：OrderedDict，存储用于处理每个观测键的网络类名称。
3. obs_nets_kwargs：OrderedDict，每个网络类的参数
4. obs_share_mods：OrderedDict，共享网络的键
5. obs_nets：nn.ModuleDict()，每个观测键的网络实例
6. obs_randomizers：nn.ModuleDict()，随机扰动器
7. feature_activation：激活函数
8. _locked：标志位，用于防止在网络构建完成后再添加新键。

## register_obs_key方法
该方法用于注册观测键。观测键由名字、形状、网络名称、网络参数、网络实例、随机数扰动与共享网络实例（可选），并将这些信息存储到类的属性中。
## create_layers方法
该方法根据之前注册的观测键信息，为每个观测键创建对应的网络，并设置好激活函数。
## make方法
该方法调用create_layers方法创建网络和激活函数，然后锁定状态。（self._locked = True）
## forward方法
该方法负责将输入观测数据字典obs_dict按键的顺序处理，并输出一个扁平化的特征张量。
obs_dict：字典，键为obs_shapes中注册的观测键，值为对应键的张量，包含批量观测数据，形状与 obs_shapes 中定义的形状一致。该方法的具体步骤如下：
1. 准备工作：确保状态已锁定（因为只有make中调用了lock，即确保先使用make）、检查输入的obs_dict是否包含所有的观测键。
2. 初始化空列表feats，用于储存每个键的处理结果
3. 遍历self.obs_shapes中的键，按顺序从obs_dict提取对应键的值x
4. 如果该键有随机扰动器，则对 x 进行扰动处理
5. 如果该键有编码器，则将x传给编码器网络；如果有激活函数则在网络输出后应用激活函数
6. 再次进行随机扰动
7. 将x展平为[B,D]
8. 将所有的特征张量沿最后一个维度拼接成一个大特征张量并返回。
## outshape方法
根据网络和随机扰动器的处理规则，计算编码器的输出维度。计算过程与forward方法类似。
## __repr__方法
该方法可以将encoder中的所有网络和随机扰动器的信息打印出来。

# ObservationDecoder类
该类与ObservationEncoder类对应，用于将展平后的特征解码还原为各模态的观察数据
## __init__构造函数
该构造函数接收2个参数：
1. decode_shapes：OrderedDict类型，将观测键映射到其输出形状。描述每种模态的解码输出形状。
2. input_feat_dim：输入特征的平展维度（输入特征向量的大小）
该方法首先验证参数合法性，然后存储输入特征维度和模态形状信息（self.obs_shapes），最后调用_create_layers 方法生成用于预测每种模态的线性层。
## _create_layers方法
该方法为每个模态创建一个nn.Linear层，将平展输入特征映射到该模态的输出形状。
该线性层的输入为input_feat_dim，输出为self.obs_shapes中键对应的值
## output_shape方法
返回解码器的输出形状（字典形式）
## forward方法
该方法用于从输入的平展特征feats中，生成各个模态的输出。即对每个模态，先使用对应的线性层生成预测结果，然后将结果重塑为该模态定义的输出形状。最终保存在字典output中。

# ObservationGroupEncoder类
## 简介
此类的作用是将多个观测组(observation groups)中的输入，分别通过各自的ObservationEncoder编码，最终将所有组的编码结果拼接成一个单一的特征向量。

observation_group（观测组）是该类的一个核心概念，其是由模态-形状对构成的字典，表示这个组包含的不同模态输入及其对应的形状。而此类接收一个观测组的字典observation_group_shapes作为输入，例如
```python
observation_group_shapes = OrderedDict({
    "obs": OrderedDict({"image": (3, 120, 160), "depth": (1, 120, 160)}),
    "goal": OrderedDict({"position": (3,), "orientation": (4,)}),
})
```
其中obs是一个观测组，包含两个模态：image和depth，每个模态的形状分别为(3, 120, 160)和(1, 120, 160)。而goal也是一个观测组，包含两个模态：position和orientation，每个模态的形状分别为(3,)和(4,)。

同样的，每个观测组也对应一个之前提到的ObservationEncoder类。ObservationGroupEncoder会为每个观测组分配一个独立的ObservationEncoder，并将其存储在self.nets中。在前向传播中，输入的每个观测组会通过相应的ObservationEncoder编码，最终将所有观测组的特征向量拼接。
## __init__构造函数
定义三个参数：
1. observation_group_shapes：多个观测组的字典
2. feature_activation：激活函数，默认为ReLU，用于ObservationEncoder中的编码层输出（如果为None则不输出）
3. encoder_kwargs编码器配置参数
在验证参数合法性后，调用obs_encoder_factory创建ObservationEncoder对象，并存储在self.nets中。（每个观察类的encoder对应nets[obs_group]）
## forward方法
该前向传播方法的输入为一个字典，包含多个观测组的输入，每个观测组本身也是一个模态-张量对的字典。该方法的逻辑如下：
1. 确保inputs中包含所有必要的观测组
2. 遍历observation_group_shapes中定义的每个观测组，获取观测组的输入数据斌用对应的编码器对其尽进行编码
3. 将所有编码结果按照最后一维度进行拼接

# 完整的训练逻辑
从config中加载配置->从hdf5文件中加载观测键->用观测键与config中的算法名创建模型（diffusion_policy）->创建文件加载器->调用run_epoch函数（主函数就是模型中的train_on_batch，只不过加了时间计算函数）->打印训练结果并保存模型。

创建模型步骤如下：
将观察数据用OrderedDict类型保存（创建观测键），并从config中获得编码器参数->创建ObservationGroupEncoder类型编码器->确定单手/双手操作，从而确定输出空间->创建噪声预测网络ConditionalUnet1D->将两个网络合并为一个统一的网络nets->创建noise_scheduler->保存模型结构

train_on_batch：
规定To，Ta，Tp->加载训练数据，将观察obs和目标goal从观察键中分开->用编码器将环境数据转换到特征空间并展平->对动作数据加上随机噪声->用噪声预测网络对含噪数据的噪声进行预测（臂手分开），计算噪声的L2损失->反向传播


训练完成后使用run_trained_agent.py即可测试

预计下周可以对模型进行初步训练。


# LEAP Hand控制API
这周把LEAP Hand的API代码简单看了一遍。涉及到底层的代码没怎么仔细看，总结了一些顶层控制代码

set_leap(self, pose): 接收一个 LEAP 手势的位姿，并直接控制机器人手指的动作。pose 是一个包含 16 个元素的数组，表示手指各关节的角度。该函数将 pose 更新到 curr_pos 并写入到机械手。

set_ones(self, pose): 用于模拟兼容性控制，假设输入的pose范围是 [-1, 1]，并将其转换为 LEAP 手势的范围。转换后更新到 curr_pos，并发送到机械手。

read_pos(self): 从机械手读取当前的位置，即各关节的角度值。

read_vel(self): 从机械手读取当前的速度。

read_cur(self): 从机械手读取当前的电流值。

pos_vel(self): 组合命令，能够同时读取位置和速度，并返回一个包含位置和速度数据的列表。

pos_vel_eff_srv(self): 组合命令，能够读取位置、速度和电流，并返回一个包含这些数据的列表。

在实际使用的过程中调用set_leap即可对LEAP手完成控制。