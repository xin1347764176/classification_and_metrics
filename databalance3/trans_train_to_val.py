#写个python脚本，在同一个文件夹/home/resch_work/mber_codes/zxx/huiyi/databalance2/中有包含photo、val、valorigin三个文件夹，
#在val文件夹中新建所有与valorigin文件夹中相同名字的文件夹，计算valorigin文件夹中每个子文件中文件数量，拷贝一半到对应val相同名字子文件中，并从train文件夹中对应相同名称的子文件夹中拷贝一半到val对应相同名称子文件中

import os
import shutil
#写个python脚本清除/home/resch_work/mber_codes/zxx/huiyi/databalance2/val/中所有文件
# 定义要清除的文件夹路径
folder_path = '/home/resch_work/mber_codes/zxx/huiyi/databalance2/val'
# 遍历文件夹中的文件，并删除它们
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)


# 定义文件夹路径
base_dir = '/home/resch_work/mber_codes/zxx/huiyi/databalance2/'
val_dir = os.path.join(base_dir, 'val')
valorigin_dir = os.path.join(base_dir, 'valorigin')
train_dir = os.path.join(base_dir, 'train')

# 创建与valorigin文件夹中相同名字的文件夹，并拷贝一半文件
for root, dirs, files in os.walk(valorigin_dir):
    for subdir in dirs:
        subdir_path = os.path.join(val_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        origin_files = os.listdir(os.path.join(valorigin_dir, subdir))
        num_files = len(origin_files)
        # needcopy = 0
        needcopy = num_files // 100

        for i, filename in enumerate(origin_files):
            if i >= needcopy:
                break

            src_file = os.path.join(valorigin_dir, subdir, filename)
            dst_file = os.path.join(subdir_path, filename)
            shutil.copyfile(src_file, dst_file)

# 从train文件夹中拷贝一半文件到相应的val子文件夹中
for root, dirs, files in os.walk(train_dir):
    for subdir in dirs:
        subdir_path = os.path.join(val_dir, subdir)

        train_files = os.listdir(os.path.join(train_dir, subdir))
        num_files = len(origin_files)
        num_files_to_copy = num_files-needcopy

        for i, filename in enumerate(train_files):
            if i >= num_files_to_copy:
                break

            src_file = os.path.join(train_dir, subdir, filename)
            dst_file = os.path.join(subdir_path, filename)
            shutil.copyfile(src_file, dst_file)

print('done')