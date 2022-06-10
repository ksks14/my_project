import cv2
import torch
# 设置GUI
import tkinter
# 播放视频
import PIL.Image, PIL.ImageTk

# get my class
from my_video import my_video
from my_threading import Mythread

from utils.datasets import LoadImages
from utils.utils import non_max_suppression, scale_coords, load_classes, plot_one_box

from traffic_light.models import Darknet


class Myapp:
    def __init__(self, window, title, frame, vid_source=''):
        # set the plt colors
        self.colors = [(0, 255, 0), (0, 0, 255), (0, 0, 155), (0, 200, 200), (29, 118, 255), (0, 118, 255)]
        # load the cuda
        self.device = torch.device('cuda:0')
        # load the traffic model 
        self.tra_model = None
        self.tra_names = None
        self.load_tra_model()
        # load the video
        self.delay = 15
        self.vid = my_video(vid_source=vid_source)
        # get the frame width and height
        # self.vid_width = self.vid.width
        # self.vid_height = self.vid.height
        self.vid_width = 612
        self.vid_height = 612
        # set the window
        self.window = window
        self.window.title = title
        # self.window_wid make the window in the middle of the screen
        self.window_wid = (self.vid_width + 100) * 2
        self.window.geometry("%dx%d+%d+%d" % (
            self.window_wid, self.vid_height, (window.winfo_screenwidth() - self.window_wid) // 2,
            (window.winfo_screenheight() - self.vid_height) / 2))
        # add the frame
        self.add_frame_to_window(frame)
        # add the canvas to show the frame
        self.add_canvas_to_window()
        # into the process
        # 该方法跳转至模型处理部分与多线程部分
        self.updata()
        # run
        self.window.mainloop()

    def add_frame_to_window(self, frame):
        """

        :param frame:
        :return:
        """
        # 根据父组件创建
        my_frame = frame(self.window)
        # 布局
        my_frame.pack(side='left', padx=10)
        tkinter.Button(my_frame, text="暂停", height=2, width=10, command=self.test_job).pack(side="top")
        tkinter.Button(my_frame, text="恢复", height=2, width=10, command=self.test_job).pack(side="top")
        # tkinter.Button(my_frame, text="结束线程", height=2, width=10, command=self.test_job).pack(side="top")
        tkinter.Button(my_frame, text="退出程序", height=2, width=10, command=self.window.quit).pack(side="top")

    def test_job(self):
        """

        :return:
        """
        pass

    def add_canvas_to_window(self):
        """

        :return:
        """
        self.canvas1 = tkinter.Canvas(self.window, bg="#c4c2c2", height=self.vid_height, width=self.vid_width)
        self.canvas1.pack(side='left')
        self.canvas2 = tkinter.Canvas(self.window, bg="#c4c2c2", height=self.vid_height, width=self.vid_width)
        self.canvas2.pack(side='right')

    def updata(self):
        """

        :return:
        """
        # get the ground truth shape:(480, 640, 3)
        frame = self.vid.get_frame()
        res_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 以下为处理过程，为了保证速度，需要在整个tkinter一开始就加载好模型。
        res = Mythread(self.process, args=(res_frame, (self.vid_width, self.vid_height)))
        res.start()
        res.join()
        self.photo_src = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.resize(frame, (612, 612))))
        self.photo_dst = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(res.get_result()))
        self.canvas1.create_image(0, 0, image=self.photo_src, anchor=tkinter.NW)
        self.canvas2.create_image(0, 0, image=self.photo_dst, anchor=tkinter.NW)
        self.window.after(self.delay, self.updata)

    def process(self, frame=None, old_size=None):
        """

        :return:
        """
        datasets = LoadImages(frame=frame)
        pred = self.tra_process(datasets)
        try:
            # return cv.flip(frame, 1)
            return pred
        except Exception as e:
            raise ValueError(str(e))

    def load_tra_model(self):
        """
        load the tra module
        :return: 
        """
        img_size = 512
        # 加载cuda
        device = torch.device('cuda:0')
        # 加载模型并转化到GPU
        model = Darknet('../traffic_light/cfg/yolov3-spp-6cls.cfg', img_size)
        model.load_state_dict(torch.load('../traffic_light/weights/best_model_12.pt', map_location=device)['model'])
        model.to(device).eval()
        self.tra_model = model
        self.tra_names = load_classes('../traffic_light/data/traffic_light.names')

    def tra_process(self, datasets):
        with torch.no_grad():
            for img, img0 in datasets:
                im0 = img0
                img = torch.from_numpy(img).to(self.device)
                img = img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = self.tra_model(img)[0]
                pred = non_max_suppression(pred, 0.3, 0.6,
                                           multi_label=False, )
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                        for *xyxy, conf, cls in det:
                            if True:  # Add bbox to image
                                # label = '%s %.2f' % (names[int(cls)], conf)
                                label = '%s' % (self.tra_names[int(cls)])
                                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])
                return cv2.resize(im0, (612, 612), )


if __name__ == '__main__':
    file_path = '../test_data/mp4/test.mp4'
    app = Myapp(window=tkinter.Tk(), title='test', frame=tkinter.Frame, vid_source=file_path)
