# import some libs
import cv2
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageEnhance
import urllib
from skimage import io

# load data
def load_dataset(image_dir):
    '''
    This function loads in images and their labels and places them in a list
    image_dir:directions where images stored
    '''
    im_list = []
    file_lists = glob.glob(os.path.join(image_dir,  '*'))
    print(len(file_lists))
    for file in file_lists:
        im = mpimg.imread(file)
        if not im is None:
            im_list.append(im)
    return im_list


def standardize(image_list):
    '''
    This function takes a rgb image as input and return a standardized version
    image_list: image and label
    '''
    standard_list = []
    # Iterate through all the image-label pairs
    for item in image_list:
        standardized_im = standardize_input(item)
        standard_list.append(standardized_im)
    return standard_list


def standardize_input(image):
    # Resize all images to be 32x32x3
    standard_im = cv2.resize(image, (32, 32))
    return standard_im


def estimate_label(rgb_image,display=False):
    '''
    rgb_image:Standardized RGB image
    '''
    return red_green_yellow(rgb_image,display)
def findNoneZero(rgb_image):
    rows,cols, a= rgb_image.shape

    counter = 0
    for row in range(rows):
        for col in range(cols):
            pixels = rgb_image[row,col]
            if sum(pixels)!=0:
                counter = counter+1
    return counter
def red_green_yellow(rgb_image,display):
    '''
    Determines the red , green and yellow content in each image using HSV and experimentally
    determined thresholds. Returns a Classification based on the values
    '''
    hsv = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)
    sum_saturation = np.sum(hsv[:,:,1])# Sum the brightness values
    area = 3232
    avg_saturation = sum_saturation / area #find average

    sat_low = int(avg_saturation*1.3) #均值的1.3倍，工程经验
    val_low = 140
    #Green
    lower_green = np.array([70,sat_low,val_low])
    upper_green = np.array([100,255,255])
    green_mask = cv2.inRange(hsv,lower_green,upper_green)
    green_result = cv2.bitwise_and(rgb_image,rgb_image,mask = green_mask)
    # White
    lower_white = np.array([0, sat_low, val_low])
    upper_white = np.array([175, 255, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_result = cv2.bitwise_and(rgb_image, rgb_image, mask=white_mask)
    #Yellow
    lower_yellow = np.array([10,sat_low,val_low])
    upper_yellow = np.array([60,255,255])
    yellow_mask = cv2.inRange(hsv,lower_yellow,upper_yellow)
    yellow_result = cv2.bitwise_and(rgb_image,rgb_image,mask=yellow_mask)

    # Red
    lower_red = np.array([150,sat_low,val_low])
    upper_red = np.array([180,255,255])
    red_mask = cv2.inRange(hsv,lower_red,upper_red)
    red_result = cv2.bitwise_and(rgb_image,rgb_image,mask = red_mask)


    sum_green = findNoneZero(green_result)
    sum_red = findNoneZero(red_result)
    sum_yellow = findNoneZero(yellow_result)
    sum_white = findNoneZero(white_result)
   # if sum_white>=sum_red and sum_white>=sum_yellow and sum_white>=sum_green:
       # return "white"
    if sum_red >= sum_yellow and sum_red>=sum_green and sum_red!=0 :
        return "前方红灯"#Red
    if sum_yellow>= sum_green and sum_yellow>=sum_red and sum_yellow!=0:
        return "前方黄灯"#yellow
    if sum_green>=sum_yellow and sum_green>=sum_red and sum_green!=0 :
        return "前方绿灯"#yellow
    return " "#green

def light(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()))
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite("du.jpg", image)

    pth2="du.jpg"
    im3 = mpimg.imread(pth2)
    tu = im3[166:392, 274:437]
    img = Image.open(pth2)
    # 对比度增强
    enh_con = ImageEnhance.Contrast(img)
    contrast = 3.0
    img_contrasted = enh_con.enhance(contrast)
    img_contrasted.save("kkk.jpg")
    pth = 'kkk.jpg'
    img1 = Image.open(pth)
    temp = int(img1.size[1] / 2)
    file_lists = glob.glob(os.path.join(pth))
    im = mpimg.imread(file_lists[0])
    # cropImg = im[0:temp]
    tt = url.split(",")
    tt2 = tt[1].split("_")
    t2 = int(tt2[1])
    if (t2>=2000 and t2<=18000) or (t2>=64000 and t2<=76000):
        cropImg = im[286:440,248:425]
        if(t2>=2000 and t2<=12000):
            tu = im3[300:403,250:490]
        if(t2>=14000 and t2<=18000):
            tu = im3[23:310,370:650]

    else:
        cropImg = im[38:61,227:325]
    #cropImg = im[30:550, 793:941]  # 获取感兴趣区域
    #cv2.imwrite("./hhh9.jpg", tu)  # 保存到指定目录
    img_test = [cropImg]
    standardtest = standardize(img_test)
    for img in standardtest:
        predicted_label = estimate_label(img, display=True)
        #print('前方' + str(predicted_label))
    return tu,predicted_label

#实时
def light3(url):
    # resp = urllib.request.urlopen(url)
    # image = np.asarray(bytearray(resp.read()))
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # cv2.imwrite("du.jpg", image)
    pth2 = url
    img10 = Image.open(pth2)
    temp = int(img10.size[1] / 2)
    im3 = cv2.imread(pth2)
    tu = im3[0:temp]
    img = Image.open(pth2)
    # 对比度增强
    enh_con = ImageEnhance.Contrast(img)
    contrast = 3.0
    img_contrasted = enh_con.enhance(contrast)
    img_contrasted.save("kkk.jpg")
    pth = 'kkk.jpg'
    file_lists = glob.glob(os.path.join(pth))
    im = mpimg.imread(file_lists[0])
    cropImg = im[0:temp]
    # cropImg = im[30:550, 793:941]  # 获取感兴趣区域
    # cv2.imwrite("./hhh9.jpg", tu)  # 保存到指定目录
    img_test = [cropImg]
    standardtest = standardize(img_test)
    for img in standardtest:
        predicted_label = estimate_label(img, display=True)
        # print('前方' + str(predicted_label))
    return tu, predicted_label

