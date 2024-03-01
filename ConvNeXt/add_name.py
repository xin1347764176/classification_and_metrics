# 在/home/resch_work/mber_codes/zxx/classification_and_metrics_demo/databalance3/val/0000-0/文件夹内所有文件名字前面加上‘0+’，比如1.tiff变成0+1.tiff
import os

folder_path = r'D:\服务器共享文件\classification_and_metrics_demo\databalance3\val\0'

for filename in os.listdir(folder_path):
    new_filename = '0+' + filename
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

folder_path = r'D:\服务器共享文件\classification_and_metrics_demo\databalance3\val\1'

for filename in os.listdir(folder_path):
    new_filename = '1+' + filename
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

folder_path = r'D:\服务器共享文件\classification_and_metrics_demo\databalance3\val\2'

for filename in os.listdir(folder_path):
    new_filename = '2+' + filename
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

print('done')