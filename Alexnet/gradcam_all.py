# import json
# import os
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# from torchvision import models
# from torchvision import transforms
# from utils import GradCAM, show_cam_on_image, center_crop_img
# from model import AlexNet

# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = AlexNet(num_classes=3)
#     model_weight_path = "./AlexNet.pth"
#     model.load_state_dict(torch.load(model_weight_path, map_location=device))
#     model.eval()
    
#     target_layers = [model.features[-1]]
    
#     data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
#     img_paths = [
#         "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/train/0/0+2-047.tiff",
#         "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/train/0/0+2-046.tiff",
#         "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/val/0/0+78-034.tiff",
#         "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/val/1/1+38-103.tiff",
#         "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/train/1/1+38-105.tiff",
#         "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/train/1/1+38-110.tiff",
#         "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/photo/2/2+25-051.tiff",
#         "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/photo/2/2+52-032.tiff",
#     ]

#     for img_path in img_paths:
#         assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
#         img = Image.open(img_path).convert('RGB')
#         plt.figure(figsize=(10, 5))
        
#         ax1 = plt.subplot(1, 2, 1)
#         ax1.set_title("Original Image")
#         plt.imshow(img)
        
#         img = np.array(img, dtype=np.uint8)
#         img_tensor = data_transform(img)
#         input_tensor = torch.unsqueeze(img_tensor, dim=0)
        
#         cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
#         grayscale_cam = cam(input_tensor=input_tensor)
#         grayscale_cam = grayscale_cam[0, :]
#         visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

#         ax2 = plt.subplot(1, 2, 2)
#         ax2.set_title("CAM Visualization")
#         plt.imshow(visualization)
        
#         plt.show()


# if __name__ == '__main__':
#     main()





import json
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=3)
    model_weight_path = "./AlexNet.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    target_layers = [model.features[-1]]
    
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    img_dir = "../databalance3/val/0/"
    
    # img_dir = "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/photo/split/0/split_3"
    # img_dir = "/home/resch_work/mber_codes/zxx/tt/test/datadelete4/photo/split/012/split_0"
    img_files = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, file) for file in img_files]

    for img_path in img_paths:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        
        file_name = img_path.split("/")[-1]
        label = file_name.split("+")[0]
        print(label + "/" + file_name)

        plt.figure(figsize=(10, 5))
        
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title("Original Image")
        plt.imshow(img)
        
        img = np.array(img, dtype=np.uint8)
        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)
        
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title("CAM Visualization")
        plt.imshow(visualization)
        
        plt.show()


if __name__ == '__main__':
    main()