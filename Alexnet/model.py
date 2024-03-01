import torch.nn as nn
import torch


# class AlexNet(nn.Module):
#     def __init__(self, num_classes=2, init_weights=False):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(      #网络层数比较多时，采用sequential，就不用“self.”
#             nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
#             #tuple：（1，2） 1代表上下方各补一行0，2代表左右两侧各补两列0
#             #nn.ZeroPad2d((1,2,1,2)) 左侧补一列，右侧补两列，上方补一行，下方补两行
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
#             nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
#             nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
#             nn.ReLU(inplace=True),
#             # nn.AdaptiveMaxPool2d(output_size=(3,2))
#             nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),                        #以0.5的概率消除节点（失活神经元）
#             nn.Linear(128 * 6 * 6, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, num_classes),
#         )
#         if init_weights:           #只是介绍一下初始化方法，一般自动使用
#             self._initialize_weights()

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, start_dim=1)    #batch channel  从channel 展平
#         # x = torch.flatten(x,1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():       #self.modules会迭代每一个定义的层结构
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:         #不为空就是0
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)      #正态分布average 0 方差0.01
#                 nn.init.constant_(m.bias, 0)


# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(AlexNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
#         self.relu4 = nn.ReLU(inplace=True)
#         self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.relu5 = nn.ReLU(inplace=True)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.dropout = nn.Dropout()
#         self.fc1 = nn.Linear(256 * 6 * 6, 4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, num_classes)
 
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = self.conv5(x)
#         x = self.relu5(x)
#         x = self.maxpool3(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.dropout(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

class AlexNet(nn.Module):
    
    def __init__(self, num_classes: int = 1000,p=0.5) -> None:
        super(AlexNet, self).__init__()
        self.p=p
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        print(f"p={p}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x