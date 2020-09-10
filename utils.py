# -*- coding: UTF-8 -*-
from cv2 import cv2 as cv
import os
import sys
from pathlib import Path
from Pose.pose_visualizer import TfPoseVisualizer

file_path = Path.cwd()
out_file_path = Path(file_path / "test_out/")
# camera resolution setting
cam_width, cam_height = 1280, 720
# input size to the model
# VGG trained in 656*368; mobilenet_thin trained in 432*368 (from tf-pose-estimation)
input_width, input_height = 656, 368



# 以用户输入参数作为输入，返回cap这么一个cv.VideroCaoture的对象实例
# (python里头该咋说？js用多了，就这么讲了，大概意思就是这样)
# 就是判断是直接启用摄像头读取视频流，还是直接从本地视频文件中拿视频流
def choose_run_mode(args):
    """
    video or webcam
    """
    # global 声明一下，表示该变量属于全局，可以直接操作其值
    global out_file_path
    if args.video:
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        out_file_path = str(out_file_path / (args.video[:-4] + '_tf_out.mp4'))
    else:
        # Webcam input
        cap = cv.VideoCapture(0)
        # 设置摄像头像素值
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_height)
        out_file_path = str(out_file_path / 'webcam_tf_out.mp4')
    return cap


def load_pretrain_model(model):
    dyn_graph_path = {
        'VGG_origin': str(file_path / "Pose/graph_models/VGG_origin/graph_opt.pb"),
        'mobilenet_thin': str(file_path / "Pose/graph_models/mobilenet_thin/graph_opt.pb")
    }
    graph_path = dyn_graph_path[model]
    if not os.path.isfile(graph_path):
        raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)

    return TfPoseVisualizer(graph_path, target_size=(input_width, input_height))


# cv.VideroCaoture实例作为输入，返回一个读写器类似，并可以设置读写帧率，
# 初始化各项参数
# round返回数值的四舍五入取值
def set_video_writer(cap, write_fps=15):
    return cv.VideoWriter(out_file_path,
                          cv.VideoWriter_fourcc(*'mp4v'),
                          write_fps,
                          (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))