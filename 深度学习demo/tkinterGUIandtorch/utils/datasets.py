import glob
import os
from pathlib import Path

import cv2
import cv2 as cv
import numpy as np
from PIL import Image, ExifTags


img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


class LoadImages:  # for inference
    def __init__(self, frame, img_size=416):
        # load the img
        self.img_0 = frame
        self.img_size = img_size

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        """

        :return:
        """
        # 该函数改变了原图的维度，尺寸，这里注意img0在后续的过程中是完全没有出现的。所以img0作为原图传参
        img0 = self.img_0
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)



def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



if __name__ == '__main__':
    img = cv.imread('../test_data/img/w.png')
    img = LoadImages(img)
    a = img[0][1]
    print(a.shape)
    cv.imshow('input', a)
    cv.waitKey()
    cv.destroyAllWindows()

