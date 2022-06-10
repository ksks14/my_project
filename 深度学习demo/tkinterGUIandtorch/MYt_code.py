import time

import cv2
import torch
# 设置GUI
import tkinter
# 播放视频
import PIL.Image, PIL.ImageTk


from utils.datasets import LoadImages
from utils.utils import non_max_suppression, scale_coords, load_classes, plot_one_box

from traffic_light.models import Darknet

device = torch.device('cuda:0')

img_size = 512

model = Darknet('./traffic_light/cfg/yolov3-spp-6cls.cfg', img_size)
model.load_state_dict(torch.load('./traffic_light/weights/best_model_12.pt', map_location=device)['model'])
model.to(device).eval()

tra_names = load_classes('./traffic_light/data/traffic_light.names')
names = load_classes('./traffic_light/data/traffic_light.names')
vid = cv2.VideoCapture('./test_data/mp4/test.mp4')
colors = [(0, 255, 0), (0, 0, 255), (0, 0, 155), (0, 200, 200), (29, 118, 255), (0, 118, 255)]
while vid.isOpened():
    ret, frame = vid.read()
    datasets = LoadImages(frame=frame, img_size=img_size)
    for img, img0 in datasets:
        print('返回后', (img.shape, img0.shape))
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print('new img shape', img.shape)

        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.3, 0.6,
                                   multi_label=False, )
        for i, det in enumerate(pred):
            im0 = img0
            s = ''
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                for *xyxy, conf, cls in det:
                    if True:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s' % (names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                cv2.imshow('input', im0)
                key = cv2.waitKey(27)
    if key == 27:
        cv2.destroyAllWindows()
        vid.release()
        break