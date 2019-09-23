import threading
import time
import requests
import lane_finder
import car_finder
import cv2
import base64
import numpy as np
ENCODING = 'utf-8'
exitFlag = 0
import find_car
import chuli
import original
#录制好的视频演示
class myThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self,id,url):
        threading.Thread.__init__(self)
        self.url = url
        self.id = id
    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        if self.id == 1:
            self.res = load1(self.url)
        if self.id == 2:
            self.res = load2(self.url)
        if self.id==3:
            self.res=load3(self.url)
    def get_result(self):
         return self.res
def load1(url):
    res=[]
    file = requests.get(url)
    img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    result, str = lane_finder.run(img)
    im2 = cv2.resize(result, (500, 300), )
    cv2.imwrite('./static/yanshi/lane.png', im2)  # 写入图片
    # yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./static/yanshi/lane.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    res.append(raw_data)
    res.append(str)
    #chuli.change(str,"lane")
    return res
def load2(url):
    res=[]
    file = requests.get(url)
    img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    result, str,tishi = find_car.run(img)
    im2 = cv2.resize(result, (500, 300), )
    cv2.imwrite('./static/yanshi/car.png', im2)  # 写入图片
    # yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./static/yanshi/car.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    res.append(raw_data)
    res.append(str)
    res.append(tishi)
    # if tishi!="":
    #    chuli.change("注意前方车辆！", "car")
    return res
def load3(url):
    res=[]
    result,str=original.light(url)
    im2 = cv2.resize(result, (500, 300), )
    cv2.imwrite('./static/yanshi/original.png', im2)  # 写入图片
    # yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./static/yanshi/original.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    res.append(raw_data)
    res.append(str)
    return res
# url="https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600"
# # 创建新线程
# thread1 = myThread(1,url)
# thread2 = myThread(2,url)

# 开启线程
# thread1.start()
# thread2.start()
# thread1.join()
# res=thread1.get_result()
# print(res)
# print("Exiting Main Thread")