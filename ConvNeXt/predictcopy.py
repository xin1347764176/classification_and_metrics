import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from model import convnext_tiny as convnext
from collections import Counter

def falsePredict(alist):
    alist=alist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 读取文件夹的路径
    folder_path = "/home/resch_work/mber_codes/zxx/classification_and_metrics_demo/databalance3/val"
    folder_path1 = "/home/resch_work/mber_codes/zxx/classification_and_metrics_demo/databalance3/val/1"
    folder_path2 = "/home/resch_work/mber_codes/zxx/classification_and_metrics_demo/databalance3/val/2"
    folder_path0 = "/home/resch_work/mber_codes/zxx/classification_and_metrics_demo/databalance3/val/0"
    model = convnext(num_classes=3).to(device)
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, 3)
    weights_path = "./temp.pth"
    model.load_state_dict(torch.load(weights_path))
    data_transform = transforms.Compose(
        [transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    json_path = './class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    if alist==[]:
        file_list = [] 
        for root, dirs, files in os.walk(folder_path): 
            for file in files: 
                file_list.append(file) 
        # fileList = [os.path.join(folder_path, file) for file in fileList]  #拼接文件路径
        alist = [[0 for i in range(4)] for j in range(len(file_list))]  #初始化列表
        for i in range(len(file_list)):
            alist[i][0] = file_list[i] #第0列文件名 1列当前有没有判错 2列是连续 3列是总的错误次数
    count=0 #用来计数弄错个数
    for root, dirs,files in os.walk(folder_path):
        for dir in dirs:
            path = os.path.join(root,dir)
            for f in os.listdir(path):
                if '1+' in f:
                    # print(os.path.join(root, f))
                    # os.remove(os.path.join(root, f))
                    # 读取图片
                    img = Image.open(os.path.join(path,f)).convert("RGB")
                    img = data_transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    model.eval()
                    with torch.no_grad():
                        # predict class        
                        output = torch.squeeze(model(img.to(device))).cpu()
                        # test=output.device
                        # print(f"output.device={test}")
                        #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
                        #  numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
                        predict = torch.softmax(output, dim=0)
                        # print(f"predict={predict}")
                        predict_cla = torch.argmax(predict).numpy() 
                        # print(f"predict_cla={predict_cla}")
                        if predict_cla !=1:
                            count+=1
                            for i in range(len(alist)):
                                for j in range(len(alist[i])):
                                    if alist[i][j] == f:
                                        alist[i][j+1]=1
                                        alist[i][j+3]+=1
                                        alist[i][j+2]+=1
                                        # if alist[i][j+1]>alist[i][j+2]:
                                        #     alist[i][j+2]=alist[i][j+1]
                                        break
                        else:
                            for i in range(len(alist)):
                                for j in range(len(alist[i])):
                                    if alist[i][j] == f:
                                        alist[i][j+1]=0
                                        alist[i][j+2]=0
                                        break
                            
                            # falseLabel.append(os.path.join(path, f))
                            # print(os.path.join(root, f))
                if '2+' in f:
                    # print(os.path.join(root, f))
                    # os.remove(os.path.join(root, f))
                    # 读取图片
                    img = Image.open(os.path.join(path,f)).convert("RGB")
                    img = data_transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    model.eval()
                    with torch.no_grad():
                        # predict class        
                        output = torch.squeeze(model(img.to(device))).cpu()
                        # test=output.device
                        # print(f"output.device={test}")
                        #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
                        #  numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
                        predict = torch.softmax(output, dim=0)
                        # print(f"predict={predict}")
                        predict_cla = torch.argmax(predict).numpy() 
                        # print(f"predict_cla={predict_cla}")
                        if predict_cla !=2:
                            count+=1
                            for i in range(len(alist)):
                                for j in range(len(alist[i])):
                                    if alist[i][j] == f:
                                        alist[i][j+1]=1
                                        alist[i][j+3]+=1
                                        alist[i][j+2]+=1
                                        break
                        else:
                            for i in range(len(alist)):
                                for j in range(len(alist[i])):
                                    if alist[i][j] == f:
                                        alist[i][j+1]=0
                                        alist[i][j+2]=0
                                        break
                            # print(os.path.join(root, f))
                if '0+' in f:
                    # print(os.path.join(root, f))
                    # os.remove(os.path.join(root, f))
                    # 读取图片
                    img = Image.open(os.path.join(path,f)).convert("RGB")
                    img = data_transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    model.eval()
                    with torch.no_grad():
                        # predict class        
                        output = torch.squeeze(model(img.to(device))).cpu()
                        # test=output.device
                        # print(f"output.device={test}")
                        #如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
                        #  numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
                        predict = torch.softmax(output, dim=0)
                        # print(f"predict={predict}")
                        predict_cla = torch.argmax(predict).numpy() 
                        # print(f"predict_cla={predict_cla}")
                        if predict_cla !=0:
                            count+=1
                            for i in range(len(alist)):
                                for j in range(len(alist[i])):
                                    if alist[i][j] == f:
                                        alist[i][j+1]=1
                                        alist[i][j+3]+=1
                                        alist[i][j+2]+=1
                                        break
                        else:
                            for i in range(len(alist)):
                                for j in range(len(alist[i])):
                                    if alist[i][j] == f:
                                        alist[i][j+1]=0
                                        alist[i][j+2]=0
                                        break
                            # print(os.path.join(root, f))
    res = [i for i in alist if i[3] >= 5]
    print(f"res={res}")
    # counter = Counter([i[0][:4] for i in res])
    # print(counter)
    print(f"count={count}")
    print(len(res))

    #在len和自己设置取最小删除
    # if not os.path.exists('removedelete.txt'):
    #     open('removedelete.txt', 'w').close()
    # else:
    #     open('removedelete.txt', 'w').close()
    # if len(res)>0:
    #     with open('removedelete.txt','a') as f:
    #         f.write("  30..  ")
    #         f.write(f'{len(res)}\n')
    #     for i in range(min(len(res),30)):
    #         if '0+' in res[i][0]:
    #             temppath = os.path.join(folder_path0, res[i][0])
    #         if '1+' in res[i][0]:
    #             temppath = os.path.join(folder_path1, res[i][0])
    #         if '2+' in res[i][0]:
    #             temppath = os.path.join(folder_path2, res[i][0])
    #         with open('removedelete.txt','a') as f:
    #             f.write(f'{res[i][0]}\n')
    #         os.remove(temppath)

    return(alist)
