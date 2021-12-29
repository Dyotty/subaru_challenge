import argparse
import json
import os
import numpy as np
import visualize_disparity_lib
from PIL import Image, ImageDraw
from io import BytesIO
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

    for scene_num in range(1000):
        video_path = video_dir_path + "/" + \
            str(scene_num).zfill(3)+"/Right.mp4"
        video = cv2.VideoCapture(video_path)  # 動画の読み込み

        ann_path = ann_dir_path + "/" + str(scene_num).zfill(3)+".json"
        save_dir_path = ann_dir_path + "/visualized/"+str(scene_num).zfill(3)

        os.makedirs(save_dir_path, exist_ok=True)

        with open(ann_path, 'rb') as f:
            ann = json.load(f)

        for frame_cnt in range(len(ann["sequence"])):
            disparity_raw_path = video_dir_path + "/" + \
                str(scene_num).zfill(3)+"/disparity/" + \
                str(frame_cnt).zfill(8) + "f.raw"
            img = visualize_disparity_lib.get_colored_disparity_image_from_raw(
                disparity_raw_path)
            ret, video_frame = video.read()
            pil_image = Image.fromarray(video_frame)
            img = visualize_disparity_lib.draw_tgt_rect_from_ann(
                img, ann["sequence"][frame_cnt])

            img.putalpha(128)
            resized_pil_image = pil_image.resize((256, 105))
            resized_pil_image.putalpha(200)
            img.paste(resized_pil_image, resized_pil_image)
            img.save(save_dir_path+"/" +
                     str(frame_cnt).zfill(8) + "f.png")


if __name__ == '__main__':
    main()
