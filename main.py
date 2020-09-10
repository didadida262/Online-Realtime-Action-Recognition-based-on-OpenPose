# -*- coding: UTF-8 -*-
# 此为测试标志
from cv2 import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# 导入相关模型
estimator = load_pretrain_model('mobilenet_thin')
# estimator = load_pretrain_model('VGG_origin')
# 返回一个TfPoseVisualizer这么个类,个中方法尚待研究

# action_classifier = load_action_premodel('Action/framewise_recognition.h5')
action_classifier = load_action_premodel('Action/framewise_recognition_under_scene.h5')

# 参数初始化
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

# 读写视频文件,两种选择，直接打开摄像头实时读取，
# 或者读取本地视频
# cv.VideroCaoture： 摄像头
cap = choose_run_mode(args)

#cv.VideoWriter 视频读写器
video_writer = set_video_writer(cap, write_fps=int(7.0))

# # 保存关节数据的txt文件，用于训练过程(for training)
# f = open('origin_data.txt', 'a+')
# cv.waitKey(1),1ms后返回-1，实质就是无限循环

while cv.waitKey(1) < 0:
    # cv.waitKey(1000)
    # has_frame:boolean,应该是该帧是否读取成功？
    # show: 显然就是帧数据，可以理解一帧就是一张图片，以数组形式存储 
    has_frame, show = cap.read()

# 从这个判断基本能判定，只有在has_frame为true，即该帧有效情况下，才处理帧数据，否则丢弃（丢帧？）
    if has_frame:
        fps_count += 1
        frame_count += 1
        # pose estimation
        humans = estimator.inference(show)
        # print('humans:',humans)
        # get pose info
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
        # print('pose:',pose)
        # recognize the action framewise
        show = framewise_recognize(pose, action_classifier)

        height, width = show.shape[:2]
        # 显示实时FPS值
        if (time.time() - start_time) > fps_interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            # 帧率和帧数是两码事儿
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 帧数清零
            start_time = time.time()
        fps_label = 'FPS:{:.2f}'.format(realtime_fps)
        cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2)

        # 显示检测到的人数
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示目前的运行时长及总帧数
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow('Action Recognition based on OpenPose', show)
        video_writer.write(show)

        # # 采集数据，用于训练过程(for training)
        # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        # f.write(' '.join(joints_norm_per_frame))
        # f.write('\n')

video_writer.release()
cap.release()
# f.close()
