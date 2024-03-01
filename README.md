# 代码使用简介

本项目在Linux环境下运行。代码参考自 [https://github.com/WZMIAOMIAO/deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

1. **数据集准备**  
   首先将划分好标签的数据集放在 `databalance3\photo` 文件夹中。使用 `split_data.py` 用于按一定比例划分训练集和验证集。

2. **数据预处理**  
   使用 `ConvNeXt\add_name.py` 将验证集每个标签对应的图片名加上前缀，以便后期找出困难样本。例如，可以训练90个epoch，之后每个epoch保存权重，在每个验证集图片再次预测。若与标签不符，则记录为一次错误，从而找出连续预测10次都是错误的所有样本信息。

3. **模型训练**  
   利用 `ConvNeXt/train.py` 文件进行训练，可以保存时间、T-sne（使用 `tsne.py` 导入npy文件）、混淆矩阵（使用 `confusion.py` 直接画）等信息。通过设置 `if epoch+1>=5` 可以指定从第几步开始保存 `temp.pth` 文件进行二次预测val图片，以找出预测错误的图片信息。

4. **可视化工具**
  `(Alexnet)` 文件夹主要是可视化 grad-cam，利用 `Alexnet\traincopy.py` 训练网络，导入.pth权重。`Alexnet\gradcam_one.py` 用来指定一张图片的 grad-cam，`Alexnet\gradcam_all.py` 用来指定某个文件夹的 grad-cam。
