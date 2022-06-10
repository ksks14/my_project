from signal import signal, SIGINT, SIG_IGN, SIG_DFL
from time import sleep

def check_signal(time=10):
    """
    test the sigal lib
    :param time:
    :return:
    """
    signal(SIGINT, SIG_IGN)
    try:
        sleep(time)
    except Exception as e:
        raise ValueError('success break')

    print('cant break')


def check_arg(a, b, c):
    print(a)

if __name__ == '__main__':
    # check_signal()
    a = {'model_path': {'data': 1}}
