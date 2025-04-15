import socket
import json
import redis
import numpy as np

# Initialize Redis connection
redis_host = "localhost"
redis_port = 6379
redis_password = ""  # If your Redis server has no password, keep it as an empty string.
r = redis.StrictRedis(
    host=redis_host, port=redis_port, password=redis_password, decode_responses=True
)

# 定义左右手关节名称 拇指、食指、中指、无名指、小指。 每个手指的关节名称都按从近到远（Proximal -> Medial -> Distal -> Tip）的顺序排列。


right_hand_joint_names = ["RightHand",
                         'RightFinger1Metacarpal', 'RightFinger1Proximal', 'RightFinger1Distal', 'RightFinger1Tip',
                         'RightFinger2Proximal', 'RightFinger2Medial', 'RightFinger2Distal', 'RightFinger2Tip',
                         'RightFinger3Proximal', 'RightFinger3Medial', 'RightFinger3Distal', 'RightFinger3Tip',
                         'RightFinger4Proximal', 'RightFinger4Medial', 'RightFinger4Distal', 'RightFinger4Tip',
                         'RightFinger5Proximal', 'RightFinger5Medial', 'RightFinger5Distal', 'RightFinger5Tip']

def normalize_wrt_middle_proximal(hand_positions):#根据手中指最近的关节位置，将手部关节位置归一化
    #根据输入判断要处理左手还是右手

    middle_proximal_idx = right_hand_joint_names.index('RightFinger3Proximal')

    #获得手腕位置
    wrist_position = hand_positions[0]
    middle_proximal_position = hand_positions[middle_proximal_idx]
    bone_length = np.linalg.norm(wrist_position - middle_proximal_position)
    normalized_hand_positions = (middle_proximal_position - hand_positions) / bone_length
    return normalized_hand_positions


def start_server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Using SOCK_DGRAM for UDP
    s.bind(("192.168.199.174", port))
    print(f"Server started, listening on port {port} for UDP packets...")

    while True:
        data, address = s.recvfrom(64800)  # Receive UDP packets
        #print(data)
        decoded_data = data.decode()
        

        # Attempt to parse JSON
        try:
            received_json = json.loads(decoded_data)

            #print(received_json)

            #print(received_json["scene"]["newtons"][0]["joints"])

            # Initialize arrays to store the positions
            right_hand_positions = np.zeros((21, 3))
            right_hand_orientations = np.zeros((21, 4))

            # Iterate through the JSON data to extract hand joint positions

            for joint_name in right_hand_joint_names:
                for joint in received_json["scene"]["newtons"][0]["joints"]:
                    if joint_name in joint['name']:
                        joint_position = np.array(list(joint["position"].values()))
                        joint_rotation = np.array(list(joint["rotation"].values()))
                        right_hand_positions[right_hand_joint_names.index(joint_name)] = joint_position
                        right_hand_orientations[right_hand_joint_names.index(joint_name)] = joint_rotation


            # relative distance to middle proximal joint
            # normalize by bone distance (distance from wrist to middle proximal)
            # Define the indices of 'middleProximal' in your joint names
            right_middle_proximal_idx = right_hand_joint_names.index('RightFinger3Proximal')

            # Calculate bone length from 'wrist' to 'middleProximal' for both hands
            right_wrist_position = right_hand_positions[0]
            right_middle_proximal_position = right_hand_positions[right_middle_proximal_idx]
            right_bone_length = np.linalg.norm(right_wrist_position - right_middle_proximal_position)

            # Calculate relative positions and normalize
            normalized_right_hand_positions = (right_middle_proximal_position - right_hand_positions) / right_bone_length

            r.set("rightHandJointXyz", np.array(normalized_right_hand_positions).astype(np.float64).tobytes())
            r.set("rawRightHandJointXyz", np.array(right_hand_positions).astype(np.float64).tobytes())
            r.set("rawRightHandJointOrientation", np.array(right_hand_orientations).astype(np.float64).tobytes())


            print("\n\n")

            print("-"*50)
            print(np.round(right_hand_positions, 3))

        except json.JSONDecodeError:
            print("Invalid JSON received:")


if __name__ == "__main__":
    start_server(14043)