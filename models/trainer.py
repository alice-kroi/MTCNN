import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from modules.sampling import FaceDataset
import thop
#from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score,explained_variance_score
 
 
class Trainer:
    def __init__(self, net, save_path, dataset_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")                      #判断是否有gpu
        else:
            self.device = torch.device("cpu")
        self.net = net.to(self.device)                              #通用的属性加self
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.cls_loss_fn = nn.BCELoss()     #置信度损失函数
        self.offset_loss_fn = nn.MSELoss()   #坐标偏移量损失函数
 
        self.optimizer = optim.Adam(self.net.parameters())
 
        if os.path.exists(self.save_path):                            #是否有已经保存的参数文件
            net.load_state_dict(torch.load(self.save_path, map_location='cpu'))
        else:
            print("NO Param")
 
    def trainer(self, stop_value):
        faceDataset = FaceDataset(self.dataset_path)     #实例化对象
        dataloader = DataLoader(faceDataset, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
        loss = 0
        self.net.train()
        while True:
            loss1 = 0
            epoch = 0
            cla_label = []
            cla_out = []
            offset_label = []
            offset_out = []
            plt.ion()
            e = []
            r = []
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                img_data_ = img_data_.to(self.device)                        #得到的三个值传入到CPU或者GPU
                category_ = category_.to(self.device)
                offset_ = offset_.to(self.device)
 
                _output_category, _output_offset = self.net(img_data_)        #输出置信度和偏移值
 
 
                # print(_output_category.shape)    #torch.Size([10, 1, 1, 1])
                # print(_output_offset.shape,"=================")   #torch.Size([10, 4, 1, 1])
 
                output_category = _output_category.view(-1, 1)                  #转化成NV结构
                output_offset = _output_offset.view(-1, 4)
                # print(output_category.shape)
                # print(output_offset.shape, "=================")
 
                #正样本和负样本用来训练置信度
                category_mask = torch.lt(category_, 2)   #小于2   #一系列布尔值  逐元素比较input和other ， 即是否 \( input < other \)，第二个参数可以为一个数或与第一个参数相同形状和类型的张量。
                category = torch.masked_select(category_, category_mask)    #取到对应位置上的标签置信度   #https://blog.csdn.net/SoftPoeter/article/details/81667810
                #torch.masked_select()根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量，
 
                #上面两行等价于category_mask = category[category < 2]
                output_category = torch.masked_select(output_category, category_mask) #输出的置信度
                # print(output_category)
                # print(category)
                cls_loss = self.cls_loss_fn(output_category, category)            #计算置信度的损失
 
 
 
                offset_mask = torch.gt(category_, 0)
                offset = torch.masked_select(offset_,offset_mask)
                output_offset = torch.masked_select(output_offset,offset_mask)
                offset_loss = self.offset_loss_fn(output_offset, offset)           #计算偏移值的损失
 
                loss = cls_loss + offset_loss
                #writer.add_scalars("loss", {"train_loss": loss}, epoch)  # 标量
                self.optimizer.zero_grad()                                       #更新梯度反向传播
                loss.backward()
                self.optimizer.step()
 
                cls_loss = cls_loss.cpu().item()                                  #将损失转达CPU上计算，此处的损失指的是每一批次的损失
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()
                print("epoch:", epoch, "loss:", loss, " cls_loss:", cls_loss, " offset_loss", offset_loss)
                epoch += 1
 
                cla_out.extend(output_category.detach().cpu())
                cla_label.extend(category.detach().cpu())
                offset_out.extend(output_offset.detach().cpu())
                offset_label.extend(offset.detach().cpu())
 
                # print("cla     :")
                # print("r2       :", r2_score(cla_label, cla_out))
                # print("offset     :")
                # print("r2       :", r2_score(offset_label, offset_out))
                # print("total    :")
                # print("r2       :", r2_score(offset_label + cla_label, offset_out + cla_out))
 
                # e.append(i)                                                                            #画出r2
                # r.append(r2_score(offset_label+cla_label, offset_out+cla_out))
                # plt.clf()
                # plt.plot(e, r)
                # plt.pause(0.01)
                #
                # cla_out = list(map(int, cla_out))                                                      #map方法可以将列表中的每一个元素转为相对应的元素类型
                # cla_label = list(map(int, cla_label))
                # offset_out = list(map(int, offset_out))
                # offset_label = list(map(int, offset_label))
                #
                # print("accuracy_score :", accuracy_score(offset_label + cla_label, offset_out + cla_out))    #求的是每一批里面的
                # print("confusion_matrix :")
                # print(confusion_matrix(offset_label + cla_label, offset_out + cla_out))
                # print(classification_report(offset_label + cla_label, offset_out + cla_out))
                cla_out = []
                cla_label.clear()
                offset_out.clear()
                offset_label.clear()
                # flops, params = thop.profile_origin(self.net, (img_data_,))  # 查看参数量
                # flops, params = thop.clever_format((flops, params), format=("%.2f"))
                # print("flops:", flops, "params:", params)
                print()
 
               
            torch.save(self.net.state_dict(), self.save_path)
            #保存模型的推理过程的时候，只需要保存模型训练好的参数，
            # 使用torch.save()保存state_dict，能够方便模型的加载
            print("save success")
 
            if loss < stop_value:            
                break
 