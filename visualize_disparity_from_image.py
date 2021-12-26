import argparse
import json
import os
import numpy as np
import visualize_disparity_lib
from PIL import Image
from io import BytesIO
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    filename = '/home/sakamoto/Downloads/train_videos/000/disparity_PNG/00000000f.png'
    # mv = cv2.VideoCapture('production ID_車_主観2.mp4')  # 動画の読み込み
    # frame_count = int(mv.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画のフレームをすべて取得
    # size = (640, 480)  # サイズ指定
    # frame_rate = int(mv.get(cv2.CAP_PROP_FPS))  # 読み込んだ動画のFPS(フレームレート)を調べる
    #filename = args.filename
    visualize_disparity_lib.show_colored_disparity_image_from_image(filename)


if __name__ == '__main__':
    main()
