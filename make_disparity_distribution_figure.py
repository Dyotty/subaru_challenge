import argparse
import json
import os
import numpy as np
import visualize_disparity_lib
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import glob
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    ann_dir_path = '/home/sakamoto/Downloads/train_annotations'
    video_dir_path = "/home/sakamoto/Downloads/train_videos"

    scene_num = 0

    ann_path = ann_dir_path + "/" + str(scene_num).zfill(3)+".json"

    with open(ann_path, 'rb') as f:
        ann = json.load(f)

    for frame_cnt in range(len(ann["sequence"])):
        disparity_raw_path = video_dir_path + "/" + \
            str(scene_num).zfill(3)+"/disparity/" + \
            str(frame_cnt).zfill(8) + "f.raw"
        disparity_array = visualize_disparity_lib.get_float_disparity_nparray_from_raw(
            disparity_raw_path)
        tgt_rect = visualize_disparity_lib.get_tgt_rect_from_ann(
            ann["sequence"][frame_cnt])

        tgt_rect_int = [int(n) for n in tgt_rect]

        disparity_array_in_rect = (
            disparity_array[tgt_rect_int[0]:tgt_rect_int[2], tgt_rect_int[1]:tgt_rect_int[3]])

        plt.hist(disparity_array_in_rect.ravel())
        plt.show()


if __name__ == '__main__':
    main()
