import cv2
import numpy as np



def cut(img,box):
    #裁剪轮廓
    count = 0
    for j in box:
        for i in range(4):
            j = np.sort(j)
            x1 , y1 = j[0]
            x2 , y2 = j[2]

        img_cut = img[y1+10:y2-10 , x1+10:x2-10]   #切片裁剪图片
        cv2.imwrite(str(count) + "img.jpg" , img_cut)
        count+=1

def contour(img):
    #检测轮廓
    #返回的轮廓，每条轮廓对应属性 = cv2.findContours(图片 , 轮廓检索模式 , 轮廓近似办法)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for i in contours:
        print("轮廓数量", len(contours))
        rect = cv2.minAreaRect(i)   #生成最小外接矩阵
        boxs = cv2.boxPoints(rect)   #四个顶点坐标
        boxs = np.int0(boxs)
        print("矩阵四个顶点坐标：")
        print(boxs)

        h = abs(boxs[3,1] - boxs[1,1])
        w = abs(boxs[3,0] - boxs[1,0])
        print("宽，高：",w,h)

        boxes.append(boxs)
    return boxes

def get_red():
    src = cv2.imread("../photo/photo1/ssPage1.jpg")
    hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    low_hsv = np.array([156,43,46])  #低阀值
    high_hsv = np.array([180,255,255])   #高阀值
    mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)   #设置阀值，除去背景
    print("图片大小：")
    print( mask.shape[0],mask.shape[1])   #垂直尺寸 水平尺寸

    boxs = contour(mask)   #检测轮廓
    print(boxs)

    cut(src,boxs)


if __name__ == '__main__':
    get_red()