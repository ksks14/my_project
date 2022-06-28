
import numpy as np

from io import BytesIO

from cv2 import cvtColor, COLOR_RGB2BGR, imshow, waitKey, destroyAllWindows

from PIL import Image

from torch import from_numpy

from pathlib import Path

from aiohttp import web
from asyncio import get_event_loop, gather, shield

from signal import signal, SIGINT, SIG_IGN, SIG_DFL

from concurrent.futures import ProcessPoolExecutor
from models.common import DetectMultiBackend

from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

BASE_DIR = Path(__file__).parent.parent

HAND_MODEL_PATH = str(BASE_DIR) + '/models/model.pt'

# predefine the model
_model = None
_labels = None


def load_my_model(model_path=HAND_MODEL_PATH, device='cpu', dnn=False, half=False, data=BASE_DIR / 'models/coco.yaml',
                  imgsz=(640, 640)):
    """
    to return the model
    :param model_path:
    :param device:
    :param dnn:
    :param half:
    :param data:
    :param imgsz:
    :return:
    """
    # get the models
    device = select_device(device)
    model = DetectMultiBackend(model_path, device=device, dnn=dnn, data=data, fp16=half)
    # return the model
    return model


def load_model_with_signal(model_path: str) -> None:
    """

    :param model_path:
    :return:
    """
    # protect the ram
    signal(SIGINT, SIG_IGN)
    global _model
    global _labels
    if _model is None:
        # get a models
        _model = load_my_model(model_path=model_path)


def clean_model_with_signal() -> None:
    """
    clean the model
    :return:
    """
    signal(SIGINT, SIG_DFL)
    global _model
    _model = None


async def init_models(app: web.Application, max_workers=1) -> ProcessPoolExecutor:
    """

    :param app:
    :return:
    """
    assert app, 'app is None!'
    # set the max executors num
    executor = ProcessPoolExecutor(max_workers=max_workers)
    # create the loop
    loop = get_event_loop()
    # get the executor
    run = loop.run_in_executor
    # set the executor in runing
    fs = [run(executor, load_model_with_signal, HAND_MODEL_PATH) for i in range(max_workers)]
    await gather(*fs)

    async def close_executor(app: web.Application, ) -> None:
        # executor
        # set the cleaning models in executor
        fs = [run(executor, clean_model_with_signal) for i in range(max_workers)]
        # protect the fs from being cancelled
        await shield(gather(*fs))
        # until all events done
        executor.shutdown(wait=True)

    app.on_cleanup.append(close_executor)
    app['executor'] = executor
    return executor



def load_img_byte(data, img_size=640, stride=32, auto=True):
    """

    :param: data->bytes
    :return:
    """

    img = Image.open(BytesIO(data))

    img_bgr = cvtColor(np.asarray(img), COLOR_RGB2BGR)

    img0 = letterbox(img_bgr, img_size, stride=stride, auto=auto)[0]

    # HWC to CHW, BGR to RGB
    img0 = img0.transpose((2, 0, 1))[::-1]
    from numpy import ascontiguousarray
    img0 = ascontiguousarray(img0)
    # img为原图,img0为改变后
    print('shape:', img0.shape)
    print('shape:', img_bgr.shape)
    return img0, img_bgr



def detect(data: bytes, model=None, device='cpu'):
    """

    :param data:
    :param model:
    :return:
    """
    if not model:
        model = _model
    stride, names, pt = model.stride, model.names, model.pt
    device = select_device(device)

    # get the img
    img = load_img_byte(data)
    im, im0s = img[0], img[1]

    im = from_numpy(im).to(device)

    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False, visualize=False)
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000
    from utils.general import non_max_suppression, scale_coords
    from utils.plots import Annotator
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    s = ''
    for i, det in enumerate(pred):
        im0 = im0s.copy()
        s += '%gx%g ' % im.shape[2:]  # print string
        from torch import tensor
        gn = tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        print(gn)
        imc = im0.copy()
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        print(s)
    from typing import Dict, Any
    from json import dumps
    data: Dict[str, Any] = {}
    data['pred'] = s
    data['success'] = True
    return dumps(data).encode('utf-8')

