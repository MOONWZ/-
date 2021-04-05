import cv2
import numpy as np
import os

def boxes_extract(img , name):
    image = cv2.imread(img)
    # 创建于原图尺寸相同的numpy数组（返回一个用0填充的数组），用来保存ROI
    ROI = np.zeros(image.shape , np.uint8)

    # 图片颜色空间转换
    image_hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)

    # 设置阀值，除去其他颜色
    low_hsv1 = np.array([0 , 43 , 46])
    high_hsv1 = np.array([10 , 255 , 255])
    low_hsv2 = np.array([156 , 43 , 46])
    high_hsv2 = np.array([180 , 255 , 255])
    binary1 = cv2.inRange(image_hsv , lowerb=low_hsv1 , upperb=high_hsv1)
    binary2 = cv2.inRange(image_hsv , lowerb=low_hsv2 , upperb=high_hsv2)
    binary3 = binary1 + binary2
    # 均值波滤 卷积操作 去除噪声
    binary = cv2.blur((binary3) , (1,1))

    # 轮廓检测
    # 返回的轮廓 ， 每个轮廓对应的属性值 = cv2.findContours(图片 ， 轮廓检索模式 ， 轮廓近似办法)
    contours , hierarchy = cv2.findContours(binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    print("检测到红框的轮廓：")
    print(contours)
    # 处理每个轮廓
    for i in range(len(contours)):
        # 计算轮廓周长，arcLength(图像轮廓 , 轮廓是闭合的) epsilon准确率参数，表示实际轮廓到近似轮廓的最大距离
        eplision = 0.9 * cv2.arcLength(contours[i] , True)
        # 将轮廓逼近为近似边数较少的形状，边数由epsilon确定
        approx = cv2.approxPolyDP(contours[i] , eplision , True)
        mm = cv2.moments(contours[i])
        # 计算轮廓中心位置cx,cy
        if mm['m00'] != 0:
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            color = image[cy][cx]
            area = cv2.contourArea(contours[i])

            # 分析图形几何形状
            corners = len(approx)
            if corners <=4 and (color[2] >=10 or color[0] >=10) and area >2000:
                # 最小外接矩阵
                rect = cv2.minAreaRect(contours[i])
                # 矩阵的四个顶点
                box = cv2.boxPoints(rect)
                box = np.array([np.array(i , np.int) for i in box])
                rbx = max(box[i][0] for i in range(4))
                rby = max(box[i][1] for i in range(4))
                lux = min(box[i][0] for i in range(4))
                luy = min(box[i][1] for i in range(4))
                print(rbx,lux,rby,luy)
                # 裁剪区域
                crop = image[luy:rby , lux:rbx]

                # 创建文件夹并保存裁剪区域
                if not os.path.exists('./crop'):
                    os.mkdir('./crop')
                if not os.path.exists('./crop/' + name):
                    os.mkdir('./crop/' + name)
                cv2.imwrite('./crop/' + name + './crop' + str(i).zfill(2) + ".jpg" , crop)
                # 在ROI空画布上画出轮廓，最后参数-1位轮廓线条宽度，负数直接填充区域
                cv2.drawContours(ROI , [box] , 0 , (255,255,255) , -1)
    return name,image

# 在目标文件夹获取图片裁剪
def get_photo(path):
    photolist = os.listdir(path)
    print(photolist)
    # 遍历每个文件夹
    for x in photolist:
        # 目标文件夹下的所有文件夹
        photo_path = os.path.join(path , x)
        # 该文件夹下的所有图片
        filelist = os.listdir(photo_path)
        print(photo_path)
        print(filelist)

        # 遍历每张图片
        for i in range(len(filelist)):
            smallpath = photo_path + '/'
            img_ref = smallpath + filelist[i]
            name = filelist[i][:-4]
            boxes_extract(img_ref , name)
            print("----------------" + img_ref + "裁剪成功-----------------")


if __name__ == '__main__':
    path = './photo/'
    get_photo(path)