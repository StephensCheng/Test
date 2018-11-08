# dataset的制作是训练的重点
import linecache
import torch
import os
import numpy as np
import PIL
from torch.utils.data import Dataset
from PIL import Image
import PIL.ImageOps

"""
torch.utils.data.Dataset 是一个表示数据集的抽象类.
你自己的数据集一般应该继承``Dataset``, 并且重写下面的方法:
    1. __len__ 使用``len(dataset)`` 可以返回数据集的大小
    2. __getitem__ 支持索引, 以便于使用 dataset[i] 可以 获取第i个样本(0索引)
"""


class MyDataset(Dataset):  # Dataset是torch.utils.data的父类

    def __init__(self, txt, transform=None, target_transform=None, should_invert=False):

        self.transform = transform  # 是否对数据进行处理变换，数据类型是实数型的值，传入的不是一个布尔型的值
        self.target_transform = target_transform
        self.should_invert = should_invert  # 这里的should_invert是布尔值的形式
        self.txt = txt

    def __getitem__(self, index):
        """
        继承Dataset类后，必须重写一个新的方法
        返回第idx个图像及相关信息
        :param index:
        :return:
        """
        line = linecache.getline(self.txt, np.random.randint(1, self.__len__()))  # 随机从Config.txt文件中随机取出一行
        line.strip('\n')
        img0_list = os.path.split(line)  # 将img0_list分成两部分一部分为头文件，一部分为单一内部文件
        should_get_same_class = np.random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_list = linecache.getline(self.txt, np.random.randint(1, self.__len__()))
                img1_list = os.path.split(img1_list)
                if img0_list[0] == img1_list[0]:  # 这一句表示前部分文件名相同说明是同一个人
                    break
        else:
            # 直接在所有文件路径中随机选择这种选择到同一个人的概率比较小
            img1_list = linecache.getline(self.txt, np.random.randint(1, self.__len__()))
            img1_list = os.path.split(img1_list)

        # 这个地方是将选择出来的文件传入img0,img1  PS:值得注意的一点是cv2，一般是以cv2.imread()的方式打开一个文件
        # img0 = Image.open(img0_list[1])
        # img1 = Image.open(img1_list[1])
        img0_list = os.path.join(img0_list[0], img0_list[1])
        img1_list = os.path.join(img1_list[0], img1_list[1])
        img0 = Image.open(img0_list.strip('\n'))
        img1 = Image.open(img1_list.strip('\n'))
        #将img0，img1转换为灰度图
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            # PIL.ImageOps.invert()表示将图片转化为反色图片
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        # 注意一定要返回数据+标签， 这里返回一对图像+label（应由numpy转为tensor）
        return img0, img1, torch.from_numpy(np.array([int(img1_list[0] != img0_list[0])], dtype=np.float32))

    def __len__(self):
        """
        继承Dataset类后，必须重写一个新的方法
        返回数据集的大小
        :return:
        """
        fh = open(self.txt, 'r')  # 读入文件中所有行，表示所有数据的个数多少
        num = len(fh.readlines())
        fh.close()
        return num
