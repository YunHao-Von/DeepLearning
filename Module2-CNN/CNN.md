# 2.卷积神经网络  
CNN堪称计算机视觉中最为基础的网络架构。本文的目的是使用Pytorch进行一次CNN网络的实战。  
## 1.数据准备  
本数据集为food-11数据集，共有11类
Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. （面包，乳制品，甜点，鸡蛋，油炸食品，肉类，面条/意大利面，米饭，海鲜，汤，蔬菜/水果）   
+ Training set: 9866张
+ Validation set: 3430张
+ Testing set: 3347张  

图片数据不同于矩阵数据，不同图片的大小可能会不一样，因此会转化成为不同的数据矩阵。所以在训练开始前，我们应该使用torchvision.transform的Resize函数对图片矩阵进行统一的重塑性，然后再将其转化为张量进行存储。  
```
train_tfm = transforms.Compose([
    transforms.Resize((128,128)),  # 将图像矩阵全部转化为128*128矩阵
    transforms.ToTensor(),  # 将图像矩阵转化为张量进行存储
])
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图像矩阵全部转化为128*128矩阵
    transforms.ToTensor(),  # 将图像矩阵转化为张量进行存储
])
```  
制作完转化器后，使用DatasetFolder来完成图片和标签的映射，同时使用dataloader来制作数据载入器:  
```
batch_size = 128  # 设定每个批次的图片数量
train_set = DatasetFolder("Data/training",loader=lambda x:Image.open(x),extensions='jpg',
                          transform=train_tfm)  # DataFolder会自动根据文件夹生成数据集，同时指定transform确认数据转化函数
valid_set = DatasetFolder("Data/validation",loader=lambda x:Image.open(x),extensions='jpg',
                          transform=test_tfm)  # DataFolder会自动根据文件夹生成数据集，同时指定transform确认数据转化函数
test_set = DatasetFolder("Data/evaluation", loader=lambda x: Image.open(x), extensions='jpg',
                         transform=test_tfm)  # DataFolder会自动根据文件夹生成数据集，同时指定transform确认数据转化函数
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True)
```
注:在notebook中, DataLoader中如果进程开多了会报错，建议在Notebook里设置num_workers=0.  

## 2.构建模型  
本次训练任务构建的CNN使用了三轮二维卷积神经网络，输出结果再送入全连接层进行训练，激活函数选用ReLU()激活函数。  
```
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        """
        输入参数:
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        torch.nn.MaxPool2d(kernel_size,stride,padding)
        input image size:[3,128,128]
        """
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),
            
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0)
        )  # 可以用Sequential方法来将网络进行分层
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256*8*8,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLu(),
            nn.Linear(256,11)
        )
    
    
    def forward(self,x):
        """
        input x:[batch_size,3,128,128]
        output:[batch_size,11]
        """
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

```

## 3.训练模型 
```
from pickletools import optimize


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 探明计算机器
model = Classifier().to(device)  # 初始化模型
model.device = device
criterion = nn.CrossEntropyLoss()  # 选用交叉熵函数作为损失函数
optimizer = torch.optim.Adam(model.parameters(),lr=0.0003,weight_decay=1e-5)  # 定义优化器
n_epochs = 80
for epoch in range(n_epochs):
    model.train()  # 让模型进入训练模式
    train_loss = list()
    train_accs = list()
    for batch in tqdm(train_loader):
        imgs,labels = batch
        imgs = imgs.to(device)  # 数据和模型必须在一个device上
        logits = model(imgs)  # 执行模型中的forward函数
        loss = criterion(logits,labels.to(device))  # 计算本batch的损失
        optimizer.zero_grad()  # 清空优化器中的梯度
        loss.backward()  # 计算梯度
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(),max_norm=10)  # 修剪梯度
        optimizer.step()  # 使用计算好的梯度更新模型
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()  # 计算准确度
        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print("第"+str(epoch)+"轮的损失为"+str(train_loss)+",准确率为"+str(train_acc))
    """进入验证模式"""
    model.eval()  # 模型进入验证模式
    valid_loss = list()
    valid_accs = list()
    for batch in tqdm(valid_loader):
        imgs, labels = batch
        imgs = imgs.to(device)  # 数据和模型必须在一个device上
        with torch.no_grad():
            logits = model(imgs)  # 执行模型中的forward函数
        loss = criterion(logits, labels.to(device))  # 计算本batch的损失
        acc = (logits.argmax(dim=-1) == labels.to(device)
               ).float().mean()  # 计算准确度
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print("第"+str(epoch)+"轮的验证损失为"+str(valid_loss)+",验证准确率为"+str(valid_acc))
```

tqdm是一个进度条显示器，使用tqdm后，可以实时看到训练的进程。

## 4.模型保存及预测  
```
torch.save(model.state_dict(), 'Models/model.pth')  # 保存当前模型在指定路径下
"""测试模型"""
model.eval()
predictions = list()
for batch in tqdm(test_loader):
    imgs,labels = batch
    with torch.no_grad():
        logits = model(imgs.to(device))  # 执行模型中的forward函数
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
with open('predict.csv','w') as f:
    f.write("ID,Category\n")
    for i,pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")
```