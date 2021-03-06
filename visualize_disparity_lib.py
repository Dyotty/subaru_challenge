import argparse
import json
import os
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import cv2
import json

disparity_image_width = 256  # 視差画像横サイズ 	256
disparity_image_height = 105  # 視差画像縦サイズ 	105
right_image_height = 420  # 右画像縦サイズ  	420
right_image_width = 1000  # 右画像縦サイズ  	420
inf_DP = 0			# 補正パラメータ　　Frame毎のSequenceデータから読み込み


def hmax(x): #, bins=20):
    hist, bins = np.histogram(x, density=True) #, bins=20)
    delta = (bins[1]-bins[0])/2
    xr = np.linspace(np.min(bins)+delta,np.max(bins)-delta,len(bins)-1)
    return xr[np.argmax(hist)]

def get_tgt_rect_from_ann(ann):
    leftup_i = ann["TgtXPos_LeftUp"]/4
    leftup_j = ann["TgtYPos_LeftUp"]/4
    rightbottom_i = leftup_i + ann["TgtWidth"]/4
    rightbottom_j = (
        leftup_j + ann["TgtHeight"]/4)
    tup = (leftup_i, leftup_j, rightbottom_i, rightbottom_j)
    return tup


def draw_tgt_rect_from_ann(img, ann):
    draw = ImageDraw.Draw(img)
    rectcolor = (255, 0, 0)  # 矩形の色(RGB)。red
    linewidth = 4  # 線の太さ
    pos = get_tgt_rect_from_ann(ann)
    draw.rectangle(pos, outline=rectcolor)  # , width=linewidth)
    return img


def get_float_disparity_nparray_from_raw(filename: str):
    if not os.path.isfile(filename):
        print("Error: Raw file is not exist!")
        return 0
    # file = open(filename, 'r')
    # rawdata = file.seek(number)

    format_image = "/home/sakamoto/Downloads/train_videos/000/disparity_PNG/00000000f.png"
    if not os.path.isfile(format_image):
        print("Error: Format image file is not exist!")
        return 0
    img_original = Image.open(format_image)
    img = img_original.copy()
    data = img.getdata()
    #data_dst = [0] * len(data)
    data_dst = np.zeros(len(data))
    with open(filename, 'rb') as f:
        disparity_raw = f.read()

    for right_j in range(right_image_height):
        for right_i in range(right_image_width):
            # 視差画像と右画像は原点が左下と左上で違うため上下反転
            disparity_j = int((right_image_height - right_j - 1) / 4)
            disparity_i = int(right_i / 4)  		# 横座標
            disparity = disparity_raw[(
                disparity_j * disparity_image_width + disparity_i) * 2]
            disparity += disparity_raw[(disparity_j * disparity_image_width +
                                        disparity_i) * 2 + 1] / 256     # 小数視差読み込み
            # 視差を距離へ変換
            # if disparity > 0:			 # disparity =0 は距離情報がない
            #    distance = 560 / (disparity - inf_DP)

            data_dst[disparity_i + (disparity_image_height - disparity_j - 1) *
                     disparity_image_width] = disparity

    # file.close()
    # img.putdata(tuple(data_dst))
    # return img
    # img.show()
    nparray = data_dst.reshape(disparity_image_width, disparity_image_height)

    return nparray


def get_colored_disparity_image_from_raw(filename: str):
    if not os.path.isfile(filename):
        print("Error: Raw file is not exist!")
        return 0
    # file = open(filename, 'r')
    # rawdata = file.seek(number)

    format_image = "/home/sakamoto/Downloads/train_videos/000/disparity_PNG/00000000f.png"
    if not os.path.isfile(format_image):
        print("Error: Format image file is not exist!")
        return 0
    img_original = Image.open(format_image)
    img = img_original.copy()
    data = img.getdata()
    data_dst = [0] * len(data)
    with open(filename, 'rb') as f:
        disparity_raw = f.read()

    for right_j in range(right_image_height):
        for right_i in range(right_image_width):
            # 視差画像と右画像は原点が左下と左上で違うため上下反転
            disparity_j = int((right_image_height - right_j - 1) / 4)
            disparity_i = int(right_i / 4)  		# 横座標
            disparity = disparity_raw[(
                disparity_j * disparity_image_width + disparity_i) * 2]
            disparity += disparity_raw[(disparity_j * disparity_image_width +
                                        disparity_i) * 2 + 1] / 256     # 小数視差読み込み
            # 視差を距離へ変換
            # if disparity > 0:			 # disparity =0 は距離情報がない
            #    distance = 560 / (disparity - inf_DP)

            data_dst[disparity_i + (disparity_image_height - disparity_j - 1) *
                     disparity_image_width] = calc_color_map(disparity, 0, 70)

    # file.close()
    img.putdata(tuple(data_dst))
    return img
    # img.show()


def show_colored_disparity_image_from_image(filename: str):
    if not os.path.isfile(filename):
        print("Error: File is not exist!")
        return 0
    img_original = Image.open(filename)
    img = img_original.copy()
    print(img)
    # img.show()
    data = img.getdata()
    data_dst = [None] * len(data)
    width, height = img.size

    for y in range(height):
        for x in range(width):
            intensity = data[x + y * width]
            # ext = img.getextrema()
            data_dst[x + y *
                     width] = calc_color_map(intensity[0], 0, 70)  # ext[0][0], 70)  # ext[0][1]) for noise inaccuracy
    print(data_dst)
    img.putdata(data_dst)
    img.show()


def calc_color_map(v: int, vmin: int, vmax: int):
    rgb_val = [255.0, 255.0, 255.0]
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.25 * dv)):
        rgb_val[0] = 0
        rgb_val[1] = (4.0 * (v - vmin) / dv) * 255.0
    elif (v < (vmin + 0.5 * dv)):
        rgb_val[0] = 0
        rgb_val[2] = (1.0 + 4.0 * (vmin + 0.25 * dv - v) / dv) * 255.0
    elif (v < (vmin + 0.75 * dv)):
        rgb_val[0] = (4.0 * (v - vmin - 0.5 * dv) / dv) * 255.0
        rgb_val[2] = 0
    else:
        rgb_val[1] = (1.0 + 4.0 * (vmin + 0.75 * dv - v) / dv) * 255.0
        rgb_val[2] = 0

    # return map(int, rgb_val)
    return tuple(list(map(int, rgb_val)))
