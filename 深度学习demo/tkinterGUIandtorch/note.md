### 思路构建

1. 识别效果传输
   1. 图像帧传输
      1. 利用一个函数包含原图于图像处理的结果，（是否多线程？）
      2. 在tkinter内部加载视频流，将图像帧传出。传给识别类。

### GUI

1. 构建GUI页面
   1. 根据返回的图像大小进行构建页面

### 神经网络

1. 加载模型

### debug

1. 图像帧在传输过程中，size出现变化，怎么查都没发现问题啊！
   1. 解决：关键字写错了······
2. 识别出错，检查图像过程
   1. 正确过程
      1. get the BGR
      2. 传入letterbox
         1. resize
         2. copyMakeBorder
         3. return a new IMG
      3. BGR to RGB new IMG
      4. ascontiguousarray new IMG
      5. return new_img img
   2. 锁定问题
      1. letterbox
         1. imgsize传参