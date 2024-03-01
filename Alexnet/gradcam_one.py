
import os
from cv2 import mean
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
from model import AlexNet

def main():

    model = AlexNet(num_classes=3)           #5?
    # pthfile = r'./AlexNet.pth'
    # model.load(torch.load(pthfile))
    # model.load_state_dict(torch.load(pthfile))
    print(model)
    model.load_state_dict(torch.load('./AlexNet.pth'))
    # model=torch.load('AlexNet.pth',map_location=torch.device("cpu"))
    #，map_location=torch.device("cpu") or RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the
    target_layers = [model.features[-1]]
    print(target_layers)

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    # img_path = "./../data/val/1/32-028.tiff"
    # img_path = "/home/resch_work/mber_codes/zxx/huiyi/roi/roi1/6-004.tiff"
    # img_path = "/home/resch_work/mber_codes/zxx/huiyi/roi/roi1/10-061.tiff"
    # img_path = "/home/resch_work/mber_codes/zxx/huiyi/roi/roi1/71-008.tiff"
    img_path = "../databalance3/val/0/0+2-065.tiff"
    # img_path = "./bothiadd.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')

    # im=img
    # im=transforms.Resize((224,224))(im)
    # width, height = im.size
    # print("width:", width, "height:", height)

    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)

    # im=np.array(im,dtype=np.uint8)
    # img=im
    # plt.imshow(img)

    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # target_category = 1  # tabby, tabby cat

    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor)
    # grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
# __call__是类的实例方法，当实例对象被“调用”时触发，即实例对象后加上括号像函数一样调用时，该方法被调用。
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()

if __name__ == '__main__':
    main()

# %%
