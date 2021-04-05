import cv2
import numpy as np
import os
import json


def boxes_extract(img, name): # 提取ROI
    image = cv2.imread(img) # 原图图像
    ROI = np.zeros(image.shape, np.uint8)  # 创建与原图同尺寸的空numpy数组（返回来一个给定形状和类型的用0填充的数组），用来保存ROI信息
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 颜色空间转换函数，HSV空间内分离颜色
    low_hsv1 = np.array([0, 43, 46])
    high_hsv1 = np.array([10, 255, 255])
    low_hsv2 = np.array([156, 43, 46])
    high_hsv2 = np.array([180, 255, 255])

    binary1 = cv2.inRange(image_hsv, lowerb=low_hsv1, upperb=high_hsv1) # 利用cv2.inRange函数设阈值，去除背景部分（根据红色hsv上下限转为二值图像）
    binary2 = cv2.inRange(image_hsv, lowerb=low_hsv2, upperb=high_hsv2)  # 利用cv2.inRange函数设阈值，去除背景部分（根据红色hsv上下限转为二值图像）
    binary3=binary2+binary1
    binary = cv2.blur((binary3), (1, 1)) # KEY STEP: 平均卷积操作 模糊处理减少瑕疵点

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, # EXTERNAL选择最外框
                                           cv2.CHAIN_APPROX_SIMPLE)  # 查找所有轮廓信息并保存于contours数组中

    for i in range(len(contours)):  # 基于轮廓数量处理每个轮廓
        epsilon = 0.9 * cv2.arcLength(contours[i], True)  # arcLength计算轮廓周长，contours[i]表示图像轮廓，True表示目标轮廓是闭合的
                                                           # epsilon为准确率参数，表示实际轮廓到近似轮廓的最大距离（调参）
        approx = cv2.approxPolyDP(contours[i], epsilon, True)  # 把轮廓形状近似为边数较少的形状，边数由指定的epsilon决定
        mm = cv2.moments(contours[i])
        if mm['m00'] != 0:
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00']) # 计算轮廓重心位置坐标
            color = image[cy][cx]
            area = cv2.contourArea(contours[i]) # 计算轮廓面积
            # 分析几何形状
            corners = len(approx)
            if corners <= 4 and (color[2] >= 10 or color[0] >= 10) and area >2000:  # 判定条件根据ROI特点进行调整
                rect = cv2.minAreaRect(contours[i])   # 找到最小外接矩形，该矩形可能有方向
                box = cv2.boxPoints(rect)   # box是四个点的坐标
                box = np.array([np.array(i, np.int) for i in box])
                rbx = max(box[i][0] for i in range(4))
                rby = max(box[i][1] for i in range(4))
                lux = min(box[i][0] for i in range(4))
                luy = min(box[i][1] for i in range(4))
                crop = image[luy:rby, lux:rbx] # 裁剪篡改区域
                if not os.path.exists('./crop'):
                    os.mkdir('./crop')
                if not os.path.exists('./crop/' + name):
                    os.mkdir('./crop/' + name)
                cv2.imwrite('./crop/' + name + '/crop' + str(i).zfill(2) + ".jpg", crop)  # 保存裁剪区域
                cv2.drawContours(ROI, [box], 0, (255, 255, 255),
                                 -1)  # 在ROI空画布上画出轮廓，并填充白色（最后的参数为轮廓线条宽度，如果为负数则直接填充区域）
    return name, image


#获取图片进行裁剪保存
def get_photo(path):
    photolist = os.listdir(path)  # 路径下的文件夹
    print(photolist)
    for x in photolist:  # 遍历每个文件夹
        photo_path = os.path.join(path, x)  # 现在所在的文件夹路径
        filelist = os.listdir(photo_path)  # 该文件夹中所有图片列表
        print(photo_path)
        print(filelist)
        for i in range(len(filelist)):  # 遍历所有图片
            smallpath = photo_path + '/'
            img_ref = smallpath + filelist[i]
            name = filelist[i][:-4]
            boxes_extract(img_ref, name)  # 篡改
            print("------------------  " + img_ref + "裁剪成功 ---------------------")


if __name__ == "__main__":

    path = './photo/'
    get_photo(path)
