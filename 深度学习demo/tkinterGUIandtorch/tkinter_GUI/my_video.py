import cv2 as cv


class my_video:
    def __init__(self, vid_source=''):
        # 读取视频流
        self.vid = cv.VideoCapture(0 if not vid_source else vid_source)
        # 抛出错误
        if not self.vid.isOpened():
            raise ValueError('unable to open video source!', vid_source)
        # return the height and weight
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)


    def __del__(self):
        """
        关闭流
        :return:
        """
        if self.vid.isOpened():
            # 释放资源
            self.vid.release()

    def get_frame(self):
        """

        :return:
        """
        if self.vid.isOpened():
            # 读取视频流
            ret, frame = self.vid.read()
            # 成功则返回
            # return cv.cvtColor(frame, cv.COLOR_BGR2RGB) if ret else None
            return frame