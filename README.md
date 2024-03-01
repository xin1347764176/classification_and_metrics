## 代码使用简介
环境 Linux
代码参考  https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
1.首先将划分好标签的数据集放在databalance3\photo文件夹中，split_data.py用于按一定比例划分训练集和验证集
2.使用ConvNeXt\add_name.py 将验证集每个标签对应的图片名加上 前缀，便于后期找出困难样本（比如训练90epoch，之后每个epoch保存权重，对每个验证集图片再次预测，若与标签不符，记录为一次错误，可以找出连续预测10次都是错误的所有样本信息）
3.利用ConvNeXt/train.py文件进行训练，可以保存时间、T-sne（tsne.py导入npy文件）、confusion matrix（confusion.py直接画）等信息，从if epoch+1>=5: 设置从第几步开始保存temp.pth文件进行二次预测val图片，找出预测错误的图片信息
4.(Alexnet)文件夹主要是可视化grad-cam，利用Alexnet\traincopy.py训练网络，导入pth权重，Alexnet\gradcam_one.py用来指定一张图片的grad-cam，Alexnet\gradcam_all.py用来指定某个文件夹的grad-cam