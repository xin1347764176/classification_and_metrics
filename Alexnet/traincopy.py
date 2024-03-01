import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from sklearn import manifold
import itertools
from sklearn.metrics import confusion_matrix 

from torch.utils.tensorboard import SummaryWriter
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
from itertools import product
# from predictcopy import falsePredict
#忽略
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torchsummary

parameters=dict(
    epochs=[10],
    # epochs=[40,50,60,70,80,90,100],
    step_size=[10],
    # gamma=[1,.9,.8,.7,.6,.5],
    gamma=[.8],
    weight_decay = [.00001], #0代表没有
    p=[.5],
    lr=[.0001],
    batch_size=[16]
    # weight_decay=[.1,.01,.001,.0001,.00001],
    # p=[0.1,0.2,0.3,.4,.5,.6,.7,.8,.9],
    # lr=[.0001,.0002],
    # batch_size=[16,32,64,128]
)
param_values=[v for v in parameters.values()]
for epochs,step_size,gamma,weight_decay,p,lr,batch_size in product(*param_values):
    # comment=f'batch_size={batch_size} lr={lr} p={p} weight_decay={weight_decay} '
    # tb=SummaryWriter(comment=comment)
    epochs = epochs  #10------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
                                    # transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),#翻转之后对网络来说是全新的训练样例，一张图反过来放看他能不能认出来
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path   os.getcwd()获取当前目录

    image_path = os.path.join(data_root,"databalance3") 
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = batch_size
    # print(os.cpu_count())
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process '.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=nw)   #window为0，linux非0?>

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                    batch_size=batch_size, shuffle=False,
                                                    num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                            val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    # print(cla_dict)
    # print(' aab '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))     # %.5s ：表示最多输出5个字符? https://blog.csdn.net/qq_36414085/article/details/103478634
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=3,p=p)#, init_weights=Truev
    lr=lr     #0.0002
    net.to(device)

    # torchsummary.summary(net.cuda(),(3,224,224))
    
    loss_function = nn.CrossEntropyLoss()                              
    pata = list(net.parameters())
    if weight_decay == 0 :
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)
    print(f"learning rate={lr};batchsize={batch_size}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    # print(f"train_steps={train_steps}")   #training picture//batch size(+1?)

    running_correct = 0.0
    Train_loss_list = []
    # Val_loss_list = []
    Train_Accuracy_list = []
    Valid_Accuracy_list = []
    traintimelist=[]
    valtimelist=[]
    alist=[]
    for epoch in range(epochs):
        # train
        net.train()           #使用dropout方法how
        running_loss = 0.0
        running_correct = 0.0
        train_bar = tqdm(train_loader,file = sys.stdout)   #没讲到，应该与训练进度的柱状图有关  其中file = sys.stdout的意思是，print函数会将内容打印输出到标准输出流(即 sys.stdout)
        t1=time.perf_counter()
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            running_correct += outputs.argmax(dim=1).eq(labels.to(device)).sum().item()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #打印训练进度25分讲到
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,   #没讲到，应该与训练进度的柱状图有关
                                                                        epochs,                #{:.3f} python利用format方法保留三位小数的详细内容 https://www.php.cn/python-tutorials-448883.html
                                                                        loss)        # desc（‘str‘）: 传入进度条的前缀
        traintime=time.perf_counter()-t1
        print(f"traintime={traintime}")
        traintimelist.append(traintime)
        Train_correct = running_correct / train_num
        Train_Accuracy_list.append(Train_correct)
        Train_loss = running_loss /step
        Train_loss_list.append(Train_loss)
        print(time.perf_counter()-t1)#记录训练时间

        # tb.add_scalar('Train_loss',Train_loss,epoch+1)
        # tb.add_scalar('Train_Accuracy',Train_correct,epoch+1)

        # validate
        net.eval()      #no dropout
        acc = 0.0  # accumulate accurate number / epoch
        count=0
        val_loss = 0.0
        t2=time.perf_counter()
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for stp, val_data in enumerate(val_bar):
                # for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                #t-sne
                count = count+1
                if count == 1:
                    X = outputs
                    Y = val_labels
                elif count >= 2:
                    X = torch.cat((X, outputs), dim=0)
                    Y = torch.cat((Y, val_labels), dim=0)
            valtime=time.perf_counter()-t2
            print(f"valtime={valtime}")
            valtimelist.append(valtime)
            val_accurate = acc / val_num
            Valid_Accuracy_list.append(val_accurate)
            # Val_loss = val_loss /stp
            # Val_loss_list.append(Val_loss)

            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                (epoch + 1, running_loss / train_steps, val_accurate))

            # if Train_loss<=.010 and val_accurate>.95 and optimizer.param_groups[0]['lr']>1e-8:
            #     scheduler2= torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
            #     scheduler2.step()
            #     print("scheduler2")
            # elif  val_accurate>.80 and optimizer.param_groups[0]['lr']>1e-6:
            #     scheduler3= torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
            #     scheduler3.step()
            #     print("scheduler3")
            # elif val_accurate>.70 and optimizer.param_groups[0]['lr']>1e-6:
            #     scheduler31= torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            #     scheduler31.step()
            #     print("scheduler31")
            # elif optimizer.param_groups[0]['lr']<=1e-8 and val_accurate>.95 :
            #     scheduler4= torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
            #     scheduler4.step()
            #     print("scheduler4")
                
            # else:

            scheduler.step()
            print("scheduler")
            print(f"epoch={epoch} r======================{lr} {optimizer.param_groups[0]['lr']}")

            # tb.add_scalar('val_accurate',val_accurate,epoch+1)

            if epoch+1>=910:
                torch.save(net.state_dict(), 'temp.pth')
                print(f"epoch={epoch}")
                print(val_num)
                print(val_num-acc)
                #成员变量alist在上面
                alist=falsePredict(alist)
                # print(alist)

            if val_accurate > best_acc:
                best_acc = val_accurate
                best_epoch = epoch
                torch.save(net.state_dict(), save_path)
                # torch.save(net.state_dict(), 'checkpoint.pth')
                # torch.save(net.state_dict(), save_path)



    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'ro-', label="Train_loss")
    # plt.title('AlexNet')
    # plt.ylabel('Test_Loss')
    # plt.legend(loc='best')
    x1 = range(0, epochs)
    y2 = Train_Accuracy_list
    y3 = Valid_Accuracy_list
    y4=Train_loss_list
    # y5=Val_loss_list
    # print(y2)
    # print(y3)
    # print(x1)
    plt.subplot(1, 1, 1)
    plt.plot(x1, y2, 'r.-', label="Train Accuracy")
    plt.plot(x1,y3,'bx-',label = 'Valid Accuracy')
    plt.plot(x1,y4,'gx-',label = 'Train_loss')
    # plt.plot(x1,y5,'y.-',label = 'Val_loss')
    plt.text(best_epoch,Valid_Accuracy_list[best_epoch],'%.3f'%Valid_Accuracy_list[best_epoch],ha='center',va='bottom')
    plt.xlabel('vs. epoch')
    plt.ylabel('Valid Accuracy')
    plt.legend(loc='best')

    # plt.savefig(f'./picture/{batch_size}-{lr}-weight_decay{weight_decay}step_size{step_size}delete-gamma{gamma}.png')
    plt.show()
    print(f'one section done!  batch_size={batch_size} lr={lr} p={p} weight_decay={weight_decay}step_size{step_size}gamma={gamma}  best_epoch={best_epoch} best_acc={best_acc} ')
    # with open('logger.txt','a') as f:
    #     f.write(f'one section done! delete batch_size={batch_size} lr={lr} p={p} weight_decay={weight_decay} step_size{step_size}gamma={gamma} best_epoch={best_epoch} best_acc={best_acc} \n')
# # #------------------------------------------------------------------------------------------------------------------------
    # step_size=[10],
    # gamma=[1,.9,.8,.7,.6,.5],
    # weight_decay = [.00001], #0代表没有
    # p=[.5],
    # lr=[.00001,.00002,.00003],
    # batch_size=[32]


# #------------------------------------------------------------------------------------------------------------------------


    # def plot_t_sne(preds_result, labels):
    #     # 绘制t-sne聚类图结果
    #     # using t-SNE to show the result
    #     print("start t-sne!")
    #     tsne = manifold.TSNE(n_components=2, init="pca")  # random_state=501
    #     #   TSNE是一种降维算法，用于将高维数据映射到低维空间。本例中，n_components表示将高维数据降维到二维，init="pca"表示使用PCA作为初始化方法。
    #     best_preds = preds_result.cpu().numpy()
    #     X_tsne = tsne.fit_transform(best_preds)
    #     x_min, x_max = np.min(X_tsne, axis=0), np.max(X_tsne, axis=0)
    #     encoder_result1 = ((X_tsne - x_min) / (x_max - x_min))
    #     #对tsne降维后的数据X_tsne进行归一化，先求出X_tsne每一列的最大值和最小值，
    #     # 然后计算每个值与最小值之差除以最大值和最小值之差，得到归一化以后的结果。

    #     fig = plt.figure(2)
    #     idx_1 = (labels == 0)
    #     p1 = plt.scatter(encoder_result1[idx_1, 0], encoder_result1[idx_1, 1], marker='x', color='m',
    #                     label='NM', s=50)
    #     # 这段代码的功能是绘制一个散点图，标签为OM，图形颜色为紫色，x轴坐标为encoder_result1第一列，y轴坐标为encoder_result1第二列，每个散点的大小为50。其中labels == 0代表只绘制标签为0的点，idx_1为布尔型数组，表示标签为0的点的索引，
    #     # 因此encoder_result1[idx_1, 0]表示只绘制标签为0的点的x坐标，encoder_result1[idx_1, 1]表示只绘制标签为0的点的y坐标。
    #     idx_2 = (labels == 1)
    #     p2 = plt.scatter(encoder_result1[idx_2, 0], encoder_result1[idx_2, 1], marker='o', color='r',
    #                     label='OM', s=50)
    #     idx_3 = (labels == 2)
    #     p2 = plt.scatter(encoder_result1[idx_3, 0], encoder_result1[idx_3, 1], marker='+', color='c',
    #                     label='LM', s=50)

    #     plt.legend(loc='upper right')
    #     plt.xlabel("First component", fontsize=10)
    #     plt.ylabel("Second component",  fontsize=10)
    #     plt.grid(ls='--')
    #     # plt.savefig(
    #     #     './picture/t-SNE_of_AlexNet.svg')
    #     plt.show()
    #     print("finish picturing!")
    # plot_t_sne(X, Y)



    # # # --------------------------------------------------------------------------------------------------------------------------------

    
    # def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    #     if normalize:
    #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         print("Normalized confusion matrix")
    #     else:
    #         print('Confusion matrix, without normalization')
    #     print(cm)
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes, rotation=45)
    #     plt.yticks(tick_marks, classes)
    
    #     fmt = '.2f' if normalize else 'd'
    #     thresh = cm.max() / 2.
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    
    #     plt.tight_layout()
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')

    # def get_all_preds(model,loader):
    #     all_preds=torch.tensor([])#空的新pytorch张量
    #     with torch.no_grad(): 
    #         for batch in loader:
    #             images,labels=batch
    #             preds=model(images.to(device)).cpu() #TypeError: can‘t convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory fi
    #             all_preds=torch.cat(
    #             (all_preds,preds)
    #             ,dim=0)
    #     return all_preds

    # #验证标签和验证预测的图------------------------------------------------------------------------------
    # valprediction_loader=torch.utils.data.DataLoader(validate_dataset,batch_size=32)
    # net.load_state_dict(torch.load('AlexNet.pth'))   #默认应该本次训练的net
    # val_preds=get_all_preds(net,valprediction_loader)

    
    # valcm=confusion_matrix(validate_dataset.targets,val_preds.argmax(dim=1))
    
    # names=('0?','1?','2?')
    # plt.figure(figsize=(10,10))
    # plot_confusion_matrix(valcm,names)  #true

#     #------------------------------------------------------------------------------------------
    # import matplotlib.pyplot as plt
    # import plot_confusion_matrix

    # from sklearn.metrics import confusion_matrix
    # #from resources.plotcm import plot_confusion_matrix
    # @torch.no_grad()
    # def get_all_preds(model, loader):
    #     all_preds = torch.tensor([]).to(device)
    #     for batch in loader:
    #         images, labels = batch

    #         preds = model(images.to(device))
    #         all_preds = torch.cat(
    #             (all_preds, preds)
    #             , dim=0
    #         )
    #     return all_preds

    # prediction_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    # train_preds = get_all_preds(net, prediction_loader)

    # train_preds_numpy = train_preds.cpu().numpy()
    # print(train_preds_numpy)
    # cm = confusion_matrix(train_dataset.targets, train_preds_numpy.argmax(axis=1))

    # names = (
    #     'over'
    #     ,'normal'
    #     ,'incomplete'
    # )
    # plt.figure(figsize=(4,4))
    # plot_confusion_matrix.plot_confusion_matrix(cm, names)
    # plt.show()

    # # #-------------------------------写入txt----------------------------------------
    # # with open('/home/mber_codes/Junlin_Yuan/Malinjie/figure_data.txt','a') as file:
    # #     file.write('AlexNet'+'{}'.format(parameters)+'\n')
    # #     file.write(str(Valid_Accuracy_list))


#     nvidia-smi     每隔1s刷新一次   watch -n 1 nvidia-smi      training on GPU?     https://blog.csdn.net/lansebingxuan/article/details/126105299
#         建议使用      nvidia-smi -l 1      或者   nvidia-smi --loop=n      ，这个命令执行期间一直是一个进程PID
#                           top             
#                  os.cpu_count()                   Python中的方法用于获取系统中的CPU数量
    #_____________________________________
    # import xlwt
    # workbook = xlwt.Workbook()
    # sheet = workbook.add_sheet('sheet1')
    # # 将数据写入文件,i是enumerate()函数返回的序号数
    # for i,e in enumerate(Valid_Accuracy_list):
    #     sheet.write(i,0,e)
    # for i,e in enumerate(traintimelist):
    #     sheet.write(i,1,e)
    # for i,e in enumerate(valtimelist):
    #     sheet.write(i,2,e)
    # # for i in range(3):
    # #     for j in range(3):
    # #         sheet.write(i, j+5, str(valcm[i][j]))
    # # 保存文件
    # workbook.save('tmp1120.xls')
