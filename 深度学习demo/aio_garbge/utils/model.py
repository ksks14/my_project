import pathlib

import numpy as np

from os import environ
from PIL import Image
from json import dumps
from torchvision import transforms
from aiohttp import web
from concurrent.futures import ProcessPoolExecutor
from asyncio import get_event_loop, gather, shield
from signal import signal, SIGINT, SIG_IGN, SIG_DFL
from utils.loadmymodel import resnet101
from typing import Dict, Any

from torch.nn import Linear
from torch import load, unsqueeze, no_grad

BASE_DIR = pathlib.Path(__file__).parent

model_path = str(BASE_DIR) + '/model/model_best_checkpoint_resnet101.pth.tar'
lable_path = str(BASE_DIR) + '/model/dir_label.txt'

# max_workers = 5
# set the max workers
max_workers = 1

# predefine the model
_model = None
_labels = None

# environ["CUDA_VISIBLE_DEVICES"] = "0"


def softmax(x):
    """
    to make a classifier
    :param x:
    :return:
    """
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x


def padding_black(img):
    """
    to pad the img
    :param img:
    :return:
    """
    w, h = img.size
    scale = 224. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

    size_fg = img_fg.size
    size_bg = 224

    img_bg = Image.new("RGB", (size_bg, size_bg))

    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))

    img = img_bg
    return img


val_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


def load_my_model(model_path):
    """
    load the torch model
    :param model_path:
    :return:
    """
    # set the img_size

    model = resnet101(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = Linear(fc_inputs, 214)
    # load to cuda()
    # model = model.cuda()
    # trans to cpu
    model = model
    # 加载训练好的模型
    checkpoint = load(model_path)
    # load static
    model.load_state_dict(checkpoint['state_dict'])
    # 进入评估模式
    model.eval()
    return model


def load_classes():
    """

    :return:
    """
    with open(lable_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
        return list(map(lambda x: x.strip().split('\t'), labels))


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
        # get a model
        _model = load_my_model(model_path=model_path)
        _labels = load_classes()


def clean_model_with_signal() -> None:
    """
    clean the model
    :return:
    """
    signal(SIGINT, SIG_DFL)
    global _model
    _model = None


async def init_models(app: web.Application, ) -> ProcessPoolExecutor:
    """
    to init the model in app
    :return: ProcessPoolExecutor
    """
    # set the max executors num
    executor = ProcessPoolExecutor(max_workers=max_workers)
    # create the loop
    loop = get_event_loop()
    #
    run = loop.run_in_executor
    # set the executor in runing
    fs = [run(executor, load_model_with_signal, model_path) for i in range(max_workers)]
    await gather(*fs)

    async def close_executor(app: web.Application, ) -> None:
        # executor
        # set the cleaning model in executor
        fs = [run(executor, clean_model_with_signal) for i in range(max_workers)]
        # protect the fs from being cancelled
        await shield(gather(*fs))
        # until all events done
        executor.shutdown(wait=True)

    app.on_cleanup.append(close_executor)
    app['executor'] = executor
    return executor


def load_img_byte(data):
    """

    :param: data->bytes
    :return:
    """
    from io import BytesIO
    img = Image.open(BytesIO(data))
    img = img.convert('RGB')
    img = padding_black(img)
    img = val_tf(img)
    img = unsqueeze(img, 0)
    return img


def predict_garbge(data: bytes, model=None):
    """

    :param data:
    :param model:
    :return:
    """
    # get the img
    img = load_img_byte(data)
    # image = img.cuda()
    # trans to cpu
    image = img
    if not model:
        model = _model
    pred = model(image)
    pred = pred.data.cpu().numpy()[0]
    score = softmax(pred)
    pred_id = np.argmax(score)

    data: Dict[str, Any] = {}
    data['pred'] = _labels[pred_id][0]
    data['success'] = True
    return dumps(data).encode('utf-8')
