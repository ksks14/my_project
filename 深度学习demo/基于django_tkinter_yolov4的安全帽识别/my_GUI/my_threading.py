import threading


class Mythread(threading.Thread):
    """
    这里自己构建一个类去继承原来的thread
    """

    def __init__(self, func, args):
        super(Mythread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception as e:
            return None
