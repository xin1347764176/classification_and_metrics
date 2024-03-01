import os
from classroi import ClassRoi

# src_folders = ["C:\\Users\\dell\\Desktop\\prepare\\test\\{}".format(i) for i in (20,21)]
src_folders = r'../databalance3/photo/0/'
#目标文件夹 
dst_folder = r"./test" 

# for folder in src_folders:  
#     for root, dirs, files in os.walk(folder): 
#         for file in files: 
#             if os.path.splitext(file)[1] == '.tiff': 
#                 src_file = os.path.join(root, file)            
#                 print(file)
#                 ClassRoi(src_file, dst_folder,file).main()


for file in os.listdir(src_folders):
    if file.endswith(".tiff"):  # 如果文件扩展名为.tiff
        src_file = os.path.join(src_folders, file)  # 构建源文件的完整路径
        print(file)
        ClassRoi(src_file, dst_folder, file).main()  # 调用ClassRoi类的main()方法进行处理
print("finish")


