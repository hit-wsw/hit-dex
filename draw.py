import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random

import socket
import json
import redis
import numpy as np

# Initialize Redis connection
redis_host = "localhost"
redis_port = 6669
redis_password = ""  # If your Redis server has no password, keep it as an empty string.
r = redis.StrictRedis(
    host=redis_host, port=redis_port, password=redis_password, decode_responses=True
)


def gen_data(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Using SOCK_DGRAM for UDP
    s.bind(("192.168.1.101", port))
    print(f"Server started, listening on port {port} for UDP packets...")

    while True:
        data, address = s.recvfrom(64800)  # Receive UDP packets
        #print(data)
        decoded_data = data.decode()
        
        received_json = json.loads(decoded_data)
        return received_json


'''def gen_data1():
    # 随机生成每个关节的坐标数据
    joints = [
        {"name": f"LeftFinger{i}Tip", "position": {"x": random.uniform(-0.6, -0.4), "y": random.uniform(1.2, 1.3), "z": random.uniform(0.2, 0.4)}} for i in range(1, 6)
    ]
    joints += [
        {"name": f"LeftFinger{i}Distal", "position": {"x": random.uniform(-0.6, -0.4), "y": random.uniform(1.2, 1.3), "z": random.uniform(0.2, 0.4)}} for i in range(1, 6)
    ]
    joints += [
        {"name": f"LeftFinger{i}Proximal", "position": {"x": random.uniform(-0.6, -0.4), "y": random.uniform(1.2, 1.3), "z": random.uniform(0.2, 0.4)}} for i in range(1, 6)
    ]
    joints.append({"name": "LeftHand", "position": {"x": random.uniform(-0.4, -0.3), "y": random.uniform(1.2, 1.3), "z": random.uniform(0.2, 0.3)}})

    return {
        "version": "3.0",
        "fps": 30.0,
        "scene": {
            "timestamp": random.uniform(0, 100),
            "newtons": [{
                "name": "man",
                "joints": joints
            }]
        }
    }'''


# 颜色定义
colors = {
    "finger1": "red",
    "finger2": "blue",
    "finger3": "green",
    "finger4": "orange",
    "finger5": "purple",
    "left_hand": "black"
}

# 创建三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 更新图像的函数
def update(frame):
    ax.clear()  # 清空之前的绘制内容
    data = gen_data(14043)  # 生成新的实时数据

    # 绘制每个手指的关节并连接
    for finger in range(1, 6):
        finger_points = []
        
        for joint in data["scene"]["newtons"][0]["joints"]:
            if f"LeftFinger{finger}" in joint["name"]:
                position = joint["position"]
                finger_points.append((position["x"], position["y"], position["z"]))
                ax.scatter(position["x"], position["y"], position["z"], color=colors[f"finger{finger}"])
        
        # 连接手指的点
        if finger_points:
            xs, ys, zs = zip(*finger_points)
            ax.plot(xs, ys, zs, color=colors[f"finger{finger}"])
    
    # 绘制左手的点
    '''left_hand_pos = data["scene"]["newtons"][0]["joints"][-1]["position"]
    ax.scatter(left_hand_pos["x"], left_hand_pos["y"], left_hand_pos["z"], color=colors["left_hand"])'''
    
    # 设置标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Finger Joints with Connections')

# 使用 FuncAnimation 使图像实时更新
ani = FuncAnimation(fig, update, frames=100, interval=1000)  # interval 表示刷新间隔时间（以毫秒为单位）

# 显示图形
plt.show()
