#我是在windows环境中运行
import cv2
import numpy as np
from imutils import contours  # 排序操作，也可以不用。
class ClassRoi():
    def __init__(self,readplace,saveplace,filename):
        self.readplace=readplace
        self.saveplace=saveplace
        self.filename=filename
    # 绘图展示
    def cv_show(self,name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def main(self):
        # 读取一个模板图像
        img = cv2.imread(self.readplace, )
        # img = cv2.imread('Basler_acA1920-40gm__23578932__20221124_161210553_701.tiff', )
        height, width, channels = img.shape
        print(height, width,channels)
        # self.cv_show(name='raw',img=img)

        # 灰度图
        ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # self.cv_show('COLOR_BGR2GRAY',ref)

        # 二值图像
        ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
        # self.cv_show('THRESH_BINARY_INV',ref)
        # 函数用于将ref图像进行反二值化，将阈值设置为10，当像素值小于10时，赋值为255，否则赋值为0。

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradX = cv2.morphologyEx(ref, cv2.MORPH_CLOSE, rectKernel)
        # self.cv_show('MORPH_CLOSE',gradX)
        
        refCnts, hierarchy = cv2.findContours(gradX, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = refCnts
        cur_img = img.copy()
        cv2.drawContours(cur_img,refCnts,-1,(0,0,255),3) 
        # self.cv_show('drawContours',cur_img)

        global locs
        locs = []
        for (i, c) in enumerate(refCnts):
            # 计算矩形
            (x, y, w, h) = cv2.boundingRect(c)

            if (w > 50 and w < 490) and (h > 50 and h < 490):
                # 符合的留下来
                locs.append((x, y, w, h))
        # global group
        # for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # for (i, (gX, gY, gW, gH)) in enumerate(locs):
        #     group = img[gY - 8:gY + gH + 8, gX - 8:gX + gW + 8]
            # self.cv_show('roi',group)
        gX, gY, gW, gH = locs[0] #后面再加一个什么,如果遇到不好切割的样本，就用try，except过滤，或者其它方法
        group = img[gY - 8:gY + gH + 8, gX - 8:gX + gW + 8]
        cv2.imwrite('{}/{}'.format(self.saveplace,self.filename),group)

if __name__ == '__main__':
    x=ClassRoi(r'../databalance3/photo/0/2-003.tiff',r"./test","class.tiff").main()
    print("done")