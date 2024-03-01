def plot_t_sne(preds_result, labels):
    # 绘制t-sne聚类图结果
    # using t-SNE to show the result
    print("start t-sne!")
    tsne = manifold.TSNE(n_components=2, init="pca")  # random_state=501
    #   TSNE是一种降维算法，用于将高维数据映射到低维空间。本例中，n_components表示将高维数据降维到二维，init="pca"表示使用PCA作为初始化方法。
    best_preds = preds_result
    X_tsne = tsne.fit_transform(best_preds)
    x_min, x_max = np.min(X_tsne, axis=0), np.max(X_tsne, axis=0)
    encoder_result1 = ((X_tsne - x_min) / (x_max - x_min))
    #对tsne降维后的数据X_tsne进行归一化，先求出X_tsne每一列的最大值和最小值，
    # 然后计算每个值与最小值之差除以最大值和最小值之差，得到归一化以后的结果。

    fig = plt.figure(2)
    idx_1 = (labels == 0)
    p1 = plt.scatter(encoder_result1[idx_1, 0], encoder_result1[idx_1, 1], marker='o', color='b',
                    label='NM', s=50)
    # 这段代码的功能是绘制一个散点图，标签为OM，图形颜色为紫色，x轴坐标为encoder_result1第一列，y轴坐标为encoder_result1第二列，每个散点的大小为50。其中labels == 0代表只绘制标签为0的点，idx_1为布尔型数组，表示标签为0的点的索引，
    # 因此encoder_result1[idx_1, 0]表示只绘制标签为0的点的x坐标，encoder_result1[idx_1, 1]表示只绘制标签为0的点的y坐标。
    idx_2 = (labels == 1)
    p2 = plt.scatter(encoder_result1[idx_2, 0], encoder_result1[idx_2, 1], marker='x', color='r',
                    label='HD', s=50)
    idx_3 = (labels == 2)
    p2 = plt.scatter(encoder_result1[idx_3, 0], encoder_result1[idx_3, 1], marker='+', color='c',
                    label='LM', s=50)

    # plt.legend(loc='upper right')
    plt.xlabel("First component", fontsize=15)
    plt.ylabel("Second component",  fontsize=15)
    plt.xticks(fontsize=12.5)
    plt.yticks(fontsize=12.5)
    plt.legend(loc='upper right', fontsize=12.5)
    # plt.grid(ls='--')
    # plt.savefig(
    #     './picture/t-SNE_of_AlexNet.svg')
    plt.show()
    print("finish picturing!")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
X = np.load('XXRESNET50.npy')
Y = np.load('YYRESNET50.npy')

plot_t_sne(X, Y)
