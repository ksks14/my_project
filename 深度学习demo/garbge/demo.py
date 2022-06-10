import torchvision.transforms as transforms
import cv2

img = cv2.imread('./1.png')
print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)  # numpy数组格式为（H,W,C）

img_tensor = transforms.ToTensor()(img)  # tensor数据格式是torch(C,H,W)
print(img_tensor.size())
