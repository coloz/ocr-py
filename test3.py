import cv2
import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract


# 水平投影
def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0]*h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    cv2.imwrite("./temp/h.png", hProjection)
    return h_
# 垂直投影


def getVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0]*w
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # 绘制垂直平投影图像
    for x in range(w):
        for y in range(h-w_[x], h):
            vProjection[y, x] = 255
    cv2.imwrite("./temp/p.png", vProjection)
    return w_


def delPoint(image):
    pass


if __name__ == "__main__":
    # 读入原始图像
    # origineImage = cv2.imread('./c.jfif')
    # origineImage = cv2.imread('./a.jpg')
    origineImage = cv2.imread('./c.jpg')
    # 图像高与宽
    # print(origineImage.shape)
    (x, y) = origineImage.shape[0:2]
    # # 图像拉伸到统一尺寸
    resizeImg = cv2.resize(origineImage, (int(y*2000/x), 2000))
    # resizeImg=origineImage
    # 图像灰度化
    grayImage = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./temp/gray.png", grayImage)
    # 去水印
    dst = cv2.inpaint(grayImage, mask, 3, cv2.INPAINT_NS)
    # 将图片二值化
    # retval, binImage = cv2.threshold(grayImage,150,255,cv2.THRESH_BINARY_INV)
    binImage = cv2.adaptiveThreshold(
        grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)
    cv2.imwrite("./temp/binary.png", binImage)
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img=cv2.erode(img,kernel)
    binImage = cv2.dilate(binImage, kernel)
    # 图像高与宽
    (h, w) = binImage.shape
    Position = []
    # 垂直投影
    # H = getVProjection(binImage)
    # 水平投影
    H = getHProjection(binImage)
    # print(H)
    print(len(H))
    start = 0
    H_Start = []
    H_End = []
    # 根据水平投影获取垂直分割位置
    for i in range(len(H)):
        if H[i] > 120 and start == 0:
            H_Start.append(i)
            start = 1
        if H[i] <= 120 and start == 1:
            H_End.append(i)
            start = 0
    # 分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)-1):
        # 获取行图像
        cropImg = binImage[H_Start[i]:H_End[i], 0:w]
        # print(cropImg)
        # cv2.imshow('cropImg',cropImg)
        # 对行图像进行垂直投影
        W = getVProjection(cropImg)
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0
        for j in range(len(W)):
            if W[j] > 1 and Wstart == 0:
                W_Start = j
                Wstart = 1
                Wend = 0
            if W[j] <= 1 and Wstart == 1:
                W_End = j
                Wstart = 0
                Wend = 1
            if Wend == 1:
                Position.append([W_Start, H_Start[i], W_End, H_End[i]])
                Wend = 0
    # 根据确定的位置分割字符
    for m in range(len(Position)):
        if((Position[m][2] - Position[m][0]) < 10 or (Position[m][3]-Position[m][1]) < 15):
            continue
        # cv2.imwrite(
        #     './temp/%d.png'%(m), resizeImg[Position[m][1]:Position[m][3],Position[m][0]:Position[m][2]])
        cvImage = cv2.cvtColor(
            grayImage[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]], cv2.COLOR_BGR2RGB)
        cv2.imwrite('./temp/im.png', cvImage)
        word = pytesseract.image_to_string(cvImage, lang='chi_sim',config='--psm 7')
        print(word, end='')
        cv2.rectangle(resizeImg, (Position[m][0], Position[m][1]),
                      (Position[m][2], Position[m][3]), (255, 0, 0), 1)
        cv2.imwrite("./temp/contours.png", resizeImg)
        cv2.waitKey(100)
    cv2.waitKey(0)
