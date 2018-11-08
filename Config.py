#Config存入与函数相关文件的信息
import os
class Config():
    root = os.getcwd()        #os模块获取当前文件夹的路径
    root = os.path.join(root, "att_faces")    #root模块存入att_face模块的路径
    txt_root = os.path.join(root, "train.txt")   #txt_root模块获取train.txt的路径
    train_batch_size = 32                        #批训练大小为32
    train_number_epochs = 30                     #训练周期为30次
