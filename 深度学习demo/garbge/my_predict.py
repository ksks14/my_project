from torch.nn import Linear
from torch import load, unsqueeze, no_grad
from os import environ
import numpy as np

import torchvision.transforms as transforms
from PIL import Image

# my lib
from loadmymodel import resnet101

# %matplotlib inline
environ["CUDA_VISIBLE_DEVICES"] = "0"


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x


with open('dir_label.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    labels = list(map(lambda x: x.strip().split('\t'), labels))


def padding_black(img):
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


def load_img(file_path=None):
    """

    :param file_path:
    :return:
    """
    img = Image.open(file_path)
    img = img.convert('RGB')
    img = padding_black(img)
    img = val_tf(img)
    img = unsqueeze(img, 0)
    return img


if __name__ == "__main__":
    model = resnet101(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = Linear(fc_inputs, 214)
    # load to cuda()
    model = model.cuda()
    # # 加载训练好的模型
    checkpoint = load('model_best_checkpoint_resnet101.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with no_grad():
        # ---------------load img---------------
        file_path = './1.png'
        img = load_img(file_path=file_path)
        # ---------------process img---------------
        # src = img.numpy()
        # src = src.reshape(3, 224, 224)
        # src = np.transpose(src, (1, 2, 0))
        image = img.cuda()
        pred = model(image)
        pred = pred.data.cpu().numpy()[0]
        score = softmax(pred)
        pred_id = np.argmax(score)
        print('pre：', labels[pred_id][0])

