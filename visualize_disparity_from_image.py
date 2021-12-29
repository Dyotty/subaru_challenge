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
    visualize_disparity_lib.show_colored_disparity_image_from_image(filename)


if __name__ == '__main__':
    main()
