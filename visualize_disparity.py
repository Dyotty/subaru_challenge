import argparse
import json
import os
import numpy as np
from PIL import Image
from io import BytesIO


def CalcColorMap(v: int, vmin: int, vmax: int):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    filename = '/home/sakamoto/Downloads/train_videos/000/disparity_PNG/00000000f.png'
    #filename = args.filename
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
            ext = img.getextrema()
            data_dst[x + y *
                     width] = CalcColorMap(intensity[0], ext[0][0], ext[0][1])
    img.putdata(data_dst)
    img.show()


if __name__ == '__main__':
    main()
