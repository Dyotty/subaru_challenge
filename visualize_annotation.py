import argparse
import json
import os
import numpy as np
import visualize_disparity_lib
from PIL import Image, ImageDraw
from io import BytesIO
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    filename = '/home/sakamoto/Downloads/test_annotations/000.json'

    # mv = cv2.VideoCapture('production ID_車_主観2.mp4')  # 動画の読み込み
    # frame_count = int(mv.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画のフレームをすべて取得
    # size = (640, 480)  # サイズ指定
    # frame_rate = int(mv.get(cv2.CAP_PROP_FPS))  # 読み込んだ動画のFPS(フレームレート)を調べる
    #filename = args.filename
    with open(filename, 'rb') as f:
        annotation = json.load(f)  # f.read()
    # print(json.dumps(annotation, indent=2))
    print(annotation["sequence"][0]["TgtXPos_LeftUp"])

    format_image = "/home/sakamoto/Downloads/test_videos/000/disparity_PNG/00000000f.png"
    if not os.path.isfile(format_image):
        print("Error: Format image file is not exist!")
        return 0
    img = Image.open(format_image)
    draw = ImageDraw.Draw(img)
    rectcolor = (255, 0, 0)  # 矩形の色(RGB)。red
    linewidth = 4  # 線の太さ
    draw.rectangle([(annotation["sequence"][0]["TgtXPos_LeftUp"]/4, annotation["sequence"][0]["TgtYPos_LeftUp"]/4),
                    (annotation["sequence"][0]["TgtXPos_LeftUp"]/4 + annotation["sequence"][0]["TgtWidth"]/4,
                    annotation["sequence"][0]["TgtYPos_LeftUp"]/4 + annotation["sequence"][0]["TgtHeight"]/4)],
                   outline=rectcolor)  # , width=linewidth)  # 矩形の描画
    img.show()


if __name__ == '__main__':
    main()
