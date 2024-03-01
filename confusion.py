import numpy as np
import itertools
import matplotlib.pyplot as plt
 
 
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        matrix = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
        print("VGG-16")
    else:
        print('Confusion matrix, without normalization')
    plt.figure()
    # 设置输出的图片大小
    figsize = 8, 6
    figure, ax = plt.subplots(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # 设置title的大小以及title的字体
    font_title= {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 20,
                  }
    plt.title(title,fontdict=font_title)
    # plt.colorbar()

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.set_tick_params(width=0)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,)
    plt.yticks(tick_marks, classes)
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    print (labels)
    [label.set_fontname('Times New Roman') for label in labels]
    if normalize:
        fm_int = 'd'
        fm_float = '.2%'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_float),
                     horizontalalignment="center", verticalalignment='bottom',family = "Times New Roman", weight = "normal",size = 20,
                     color="white" if cm[i, j] > thresh else "black")
    else:
        fm_int = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='bottom',
                     color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    # plt.savefig('confusion_matrix.png', dpi=600, format='png')
 
cnf_matrix = np.array([[554,4,1],  
                       [6,554,0],
                       [3,1,554]
])

#------------------------------------------------------------------------------------------

attack_types = ['NM', 'HD', 'LM']
plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='FixConvNeXt')



