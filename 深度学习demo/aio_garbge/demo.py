import cv2 as cv
from io import BytesIO
from PIL import Image
from utils.model import padding_black, val_tf

file_path = './data/img/test_tra_1.png'

img = cv.imread(file_path)

success, encoded_image = cv.imencode(".png", img)
img_byte = img.tobytes()
img_bytes = encoded_image.tobytes()


img = Image.open(BytesIO(img_bytes))
img = img.convert('RGB')
img = padding_black(img)
img = val_tf(img)
img = unsqueeze(img, 0)


print(img)


