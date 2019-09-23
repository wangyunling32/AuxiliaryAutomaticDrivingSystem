import threading
import time
import requests
import lane_finder
import car_finder
from flask import request
import json
import os
# import uuid
import cv2
import base64
from aip import AipSpeech
import original
import find_car
# import chuli
import numpy as np
from playsound import playsound
ENCODING = 'utf-8'
exitFlag = 0
#摄像头实时捕捉
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
    # file = requests.get(url)
    # img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    img=cv2.imread(url)
    result, str = lane_finder.run(img)
    im2 = cv2.resize(result, (500, 300), )
    cv2.imwrite('./static/shishi/lane.png', im2)  # 写入图片
    # yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./static/shishi/lane.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    res.append(raw_data)
    res.append(str)
    return res
def load2(url):
    res=[]
    # file = requests.get(url)
    # img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    img = cv2.imread(url)
    result, str,tishi = find_car.run(img)
    im2 = cv2.resize(result, (500, 300), )
    cv2.imwrite('./static/shishi/car.png', im2)  # 写入图片
    # yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./static/shishi/car.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    res.append(raw_data)
    res.append(str)
    res.append(tishi)
    print(res)
    return res
def load3(url):
    res=[]
    result,str=original.light3(url)
    im2 = cv2.resize(result, (500, 300), )
    cv2.imwrite('./static/shishi/original.png', im2)  # 写入图片
    # yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./static/shishi/original.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    res.append(raw_data)
    res.append(str)
    return res

#百度语音接口（文字转换成语音）
def change(str,type):
  url=""
  if type=="lane":
     url="./static/lane.mp3"
  if type=="car":
      url="./static/car.mp3"
  app_id="16460025"
  api_key="ndG8dn9iksiWvT1NKfm0ly8t"
  secret_key="MUGyermisKxxvrln1D5RwZlOpq9U1syF"
  client=AipSpeech(app_id,api_key,secret_key)
  result=client.synthesis(str,'zh',1,{'vol':5})
  if not isinstance(result, dict):
      with open(url, 'wb') as f:
          f.write(result)
  # playsound("./static/auido.mp3")
# change("哈哈哈")
# app_id="16460025"
# api_key="ndG8dn9iksiWvT1NKfm0ly8t"
# secret_key="MUGyermisKxxvrln1D5RwZlOpq9U1syF"
# client=AipSpeech(app_id,api_key,secret_key)
# result=client.synthesis('前方黄灯！','zh',1,{'vol':5})
# if not isinstance(result, dict):
#       with open('./static/yellow.mp3', 'wb') as f:
#           f.write(result)
# playsound("./static/yellow.mp3")
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



