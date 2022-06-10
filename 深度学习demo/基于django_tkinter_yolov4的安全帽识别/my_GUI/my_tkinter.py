import tkinter
import cv2
import PIL
import PIL.Image, PIL.ImageTk

import numpy as np


from my_GUI.my_threading import Mythread
from utils.my_video import my_video

from my_nets.yolo import YOLO


class Myapp:
    def __init__(self, window, title, frame, vid_source=''):
        # load the model
        self.model = self.get_model()
        # set the video
        self.video_path = 0
        self.video_save_path = ""
        # load the video
        self.flag = 1
        self.delay = 10
        self.vid = my_video(vid_source=vid_source)
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


    def test_job(self):
        """

        :return:
        """
        pass

    def add_frame_to_window(self, frame):
        """

        :param frame:
        :return:
        """
        # 根据父组件创建
        my_frame = frame(self.window)
        # 布局
        my_frame.pack(side='left', padx=10)
        tkinter.Button(my_frame, text="暂停", height=2, width=10, command=self.pause).pack(side="top")
        tkinter.Button(my_frame, text="恢复", height=2, width=10, command=self.play).pack(side="top")
        # tkinter.Button(my_frame, text="结束线程", height=2, width=10, command=self.test_job).pack(side="top")
        tkinter.Button(my_frame, text="退出程序", height=2, width=10, command=self.window.quit).pack(side="top")

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
        # res_frame: cv.image
        res_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 以下为处理过程，为了保证速度，需要在整个tkinter一开始就加载好模型。
        res = Mythread(self.process, args=(res_frame, ))
        res.start()
        res.join()
        self.photo_src = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.resize(frame, (612, 612))))
        self.photo_dst = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(res.get_result()))
        self.canvas1.create_image(0, 0, image=self.photo_src, anchor=tkinter.NW)
        self.canvas2.create_image(0, 0, image=self.photo_dst, anchor=tkinter.NW)
        if self.flag:
            self.window.after(self.delay, self.updata)


    def pause(self):
        """
        :return:
        """
        self.flag = 0


    def play(self):
        """
        :return:
        """
        self.flag = 1
        self.window.after(self.delay, self.updata)

    def process(self, frame=None, old_size=None):
        """

        :param frame: cv.image
        :param old_size:
        :return:
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame: PIL.image
        frame = PIL.Image.fromarray(np.uint8(frame))
        print(type(frame))
        print('size:', frame.size)
        frame = np.array(self.model.detect_image(frame))
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            return frame
        except Exception as e:
            raise ValueError('wrong!')

    def get_model(self):
        """
        get the deep model
        :return:
        """
        return YOLO()




if __name__ == '__main__':
    file_path = 'D:/Desktop/learning_data/dachuang/static/test_data/mp4/safehat.mp4'
    Myapp(tkinter.Tk(), 'safehat', tkinter.Frame, vid_source=file_path)



