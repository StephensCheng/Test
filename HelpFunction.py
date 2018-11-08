import matplotlib.pyplot as plt
import numpy as np
from Config import Config
import os


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()  # 将照片转化为numpy()数组的形式即将torch.FloatTensor转换为numpy
    plt.axis("off")  # 不显示坐标
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})  # facecolor前景色

        # plt.imshow在现实的时候输入的是（imagesize,imagesize,channels）,
        # 而def imshow(img,text,should_save=False)中，参数img的格式为（channels,imagesize,imagesize）,
        # 这两者的格式不一致，我们需要调用一次np.transpose函数，即np.transpose(npimg,(1,2,0))，
        # 将npimg的数据格式由（channels,imagesize,imagesize）转化为（imagesize,imagesize,channels）,
        # 进行格式的转换后方可进行显示。

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


def show_plot(iteration, loss):  # 以周期和损失画出图像
    plt.plot(iteration, loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Iteration and Loss")
    plt.show()


def convert(train=True):  # 将原始文件中的文件路径写入一个train.txt的文件夹
    if (train):
        f = open(Config.txt_root, 'w')  # 以写的方式打开文件train.txt
        data_path = Config.root
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)  # 如果不存在这个文件，则在该路径下新建一个文件
        for i in range(40):
            for j in range(10):
                # img_path表示文件的每个文件名
                img_path = data_path +"\\s" + str(i + 1) + '\\' + str(j + 1) + '.pgm'
                f.write(img_path + '\n')  # 在txt文件中每一行写一个文件的路径名
        f.close()
