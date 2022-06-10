## 图像数据处理

该类方法主要用于前后端图像传输，场景如下：

1. 基本前后端传输渲染
2. 将深度学习模型包装为web接口，需要传输数据并转化推理

图像的转化一定要注意色彩空间，

### 将本地图像加载到bytes数据

```python
import cv2 as cv


def get_img_byte(file_path = './data/img/test_tra_1.png'):
    # get img data
    img = cv.imread(file_path)
    
    # trans to bytes，".png"：指定编码格式，传入一个np.array，返回是否成功与编码图像
    success, encoded_image = cv.imencode(".png", img)
    # 将编码图像转化为bytes，使用tobytes方法。
    img_bytes = encoded_image.tobytes()
    
    return img_bytes
```

### 将bytes数据转化到图像实例对象

```python
from io import BytesIO
from PIL import Image

def get_img_by_bytes(img_bytes):
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    return img

# res: <PIL.Image.Image image mode=RGB size=402x212 at 0x2938818EF70>
```

### 将pil.Image转化为ndarray

```python

import numpy as np

def img_to_np(img):
    return np.array(img)
```

### cv2转pil

**这里就存在色彩空间的变化，从BGR到RGB，opencv读取图像得到的图像是BGR的，而常用的matplotlib
，PIL.IMAGE等都是以RGB色彩空间的。**

```python
from cv2 import cvtColor, COLOR_BGR2RGB
from PIL import Image

def cv_to_pil(img):
    return Image.fromarray(cvtColor(img, COLOR_BGR2RGB))

```


### pil转cv2

```python
from numpy import array
from cv2 import cvtColor, COLOR_RGB2BGR

def pil_to_cv(pil):
    return cvtColor(array(pil), COLOR_RGB2BGR)

# res: A ndarray that supports OpencV
```


## 并行开发

**线程池是并行开发技术之一，python中并行技术还有很多，例如aio异步等等，但是不论是
如何并行，其实核心都是两个点，即多进程或者多线程，这俩种开发有自己的的适用场景**

### 线程

**多线程开发适用于一些轻量脚本类代码，例如爬虫，爬取视频，在网速支持的情况下，
可以利用多线程并行爬取多个ts文件。**

1. 线程池技术

**该类是concurrent.futures中的ThreadPoolExecutor，该lib同时也提供了进程池，如图所示![img.png](img.png)**

```python
# 代码实例可见，
from time import sleep
from concurrent.futures import ThreadPoolExecutor


def print_th(index_1, index_2):
    print(index_2)
    sleep(2)

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=10)
    for i in range(30):
        executor.submit(print_th, i, i+1)
```
**这里要了解一个工具：执行器，也叫作异步执行器，它服务与并发服务，由线程池或者进程
实例化返回。同时在事务循环中也常用，在上面的例子中，使用方法是submit()该方法可以
将线程或者进程提交到线程队列中。**

### 进程

**多进程开发适用于一些架构级开发，例如搭建一个边缘计算的系统，实现模型对于视频流的
推理，而且对模型的使用必须支持高并发，此时需要用到多进程，因为进程是资源分配和调度
的基本单位，我们把模型加载的每个进程中，才可以保证每个使用者都可以使用模型，如果多个
使用者同时访问模型，那么其与线性运行没有区别。为什么这么说。。说白话点，这个模型
是一个工具，它是用来跑数据的，一个人都占用了，另一个人怎么用···**

