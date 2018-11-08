from MyDataset import MyDataset
from torch.utils.data import DataLoader
from Config import Config
from torchvision import transforms
from torch import optim
from ContrastiveLoss import ContrastiveLoss
from HelpFunction import show_plot, convert
from SiameseNetwork import SiameseNetwork
from torch.autograd import Variable

"""
torch.utils.data中的DataLoader提供为Dataset类对象提供了:
    1.批量读取数据
    2.打乱数据顺序
    3.使用multiprocessing并行加载数据

    DataLoader中的一个参数collate_fn：可以使用它来指定如何精确地读取一批样本，
     merges a list of samples to form a mini-batch.
    然而，默认情况下collate_fn在大部分情况下都表现很好
"""
train = True
convert(train)
# from torchvision import transforms,transfrom将PIL.Image转换为tensor的方式，其次还可以进行Resize操作
# 注意这里的transfrom从目标文件中取出相关路径后，利用transfroms.Compose
train_data = MyDataset(txt=Config.txt_root, transform=transforms.Compose(
    [transforms.Resize((100, 100)), transforms.ToTensor()]), should_invert=False)
# shuffle=True将所有数据顺序打乱，num_worker表示运行多线程
train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=2, batch_size=Config.train_batch_size)

net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

# 值得注意一点在主函数文件中我们可以测试我们编的函数对不对
# 之前一直报错的原因是，我们在运行函数时使用多线程，freeze support（）
# 可以将使用多线程的函数放入主函数中运行
if __name__ == '__main__':
    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader):  # 按照键位值的方式从train_dataloader中读取数据，这里可以将train_loader视为一个列表
            img0, img1, label = data
            img0, img1, label = Variable(img0), Variable(img1), Variable(label)
            output1, output2 = net(img0, img1)
            # 运行优化过程
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            # 反向传递，自动求导
            loss_contrastive.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch:{},Counter loss:{}\n".format(epoch, loss_contrastive.data[0]))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])
    show_plot(counter, loss_history)
