import pyrealsense2 as rs
import numpy as np
import cv2

# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()

# 获取设备产品线以设置分辨率
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# 对于L515相机，我们使用以下配置
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# 启动流
pipeline.start(config)

try:
    while True:
        # 等待一组帧：深度和彩色
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # 将图像转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 应用颜色映射到深度图像（图像必须转换为每像素8位）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # 调整彩色图像大小以匹配深度图像
        color_image = cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
        
        # 水平堆叠显示图像
        images = np.hstack((color_image, depth_colormap))
        
        # 显示图像
        cv2.namedWindow('RealSense L515', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense L515', images)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()