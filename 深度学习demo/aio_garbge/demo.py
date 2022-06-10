import cv2 as cv
import numpy as np
from io import BytesIO
from PIL import Image
from utils.model import padding_black, val_tf

file_path = './data/img/test_tra_1.png'

img = cv.imread(file_path)

success, encoded_image = cv.imencode(".png", img)
img_bytes = encoded_image.tobytes()

img = Image.open(BytesIO(img_bytes))
img = img.convert('RGB')
img = np.array(img)
# img = padding_black(img)
# img = val_tf(img)


print(img)


