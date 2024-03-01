#这我是在windows中运行的
import cv2
import numpy as np
from imutils import contours  # 排序操作，也可以不用。
# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
# 读取一个模板图像
img = cv2.imread(r'../databalance3/photo/0/2-003.tiff')
height, width, channels = img.shape
print(height, width,channels)
cv_show('raw',img)

# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('COLOR_BGR2GRAY',ref)
cv2.imwrite('./test/ref.jpg',ref)

# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('THRESH_BINARY_INV',ref)
cv2.imwrite('./test/er.jpg',ref)
# 函数用于将ref图像进行反二值化，将阈值设置为10，当像素值小于10时，赋值为255，否则赋值为0。

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
gradX = cv2.morphologyEx(ref, cv2.MORPH_CLOSE, rectKernel)
cv_show('MORPH_CLOSE',gradX)
cv2.imwrite('./test/gradX.jpg',gradX)
 
refCnts, hierarchy = cv2.findContours(gradX, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = refCnts
cur_img = img.copy()
cv2.drawContours(cur_img,refCnts,-1,(0,0,255),3) 
cv_show('drawContours',cur_img)
cv2.imwrite('./test/cur_img.jpg',cur_img)


locs = []
for (i, c) in enumerate(refCnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)

    if (w > 100 and w < 400) and (h > 100 and h < 400):
        # 符合的留下来
        locs.append((x, y, w, h))
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    group = img[gY - 8:gY + gH + 8, gX - 8:gX + gW + 8]
    cv_show('roi',group)
cv2.imwrite('./test/1.jpg',group)