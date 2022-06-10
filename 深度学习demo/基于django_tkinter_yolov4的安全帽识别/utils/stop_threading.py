from inspect import isclass
from ctypes import c_long, py_object, pythonapi


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = c_long(tid)
    if not isclass(exctype):
        exctype = type(exctype)
    res = pythonapi.PyThreadState_SetAsyncExc(tid, py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    """
    to stop a thread
    :param thread:
    :return:
    """
    _async_raise(thread.ident, SystemExit)
