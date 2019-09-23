from flask import  Flask
from flask import request
import json
import os
import Hello
import cv2
import base64
import chuli
import numpy as np
import requests
import lane_finder
import car_finder
ENCODING = 'utf-8'
app = Flask(__name__)
resSets = {}
@app.route('/')           #创建路由（指url规划）
def hello_world():        #创建方法
    return
#实时监控时调用的进程
@app.route('/upload',methods=['POST','GET'])
def upload():
    f=request.files['img']
    f.save(os.path.join('./static/shishi/',"a1.png"))
    url="./static/shishi/a1.png"
    thread1 = chuli.myThread(1, url)
    thread2 = chuli.myThread(2, url)
    thread3 = chuli.myThread(3, url)
    # 开启线程
    thread1.start()
    thread2.start()
    thread3.start()
    thread1.join()
    thread2.join()
    thread3.join()
    res1 = thread1.get_result()  # 车道线
    res2 = thread2.get_result()  # 车辆检测
    res3 = thread3.get_result()#信号灯
    result=res1+res2+res3
    json_data = json.dumps(result,ensure_ascii=False)
    # print(res1)
    #print(res2)
    #print(res3)
    # print(result)
    print(json_data)
    return json_data
#安卓传过来的
# @app.route('/login',methods=['POST','GET'])
# def login():
#     print("lalall")
#     return json.dump("haha")

#使用线程
@app.route('/reload',methods=['POST','GET'])
def reload():
    time = request.args.get('time')
    url = "https://lindia.oss-cn-beijing.aliyuncs.com/002.mp4?x-oss-process=video/snapshot,t_%s,f_jpg,w_800,h_600" % time
    # 创建新线程
    thread1 = Hello.myThread(1, url)
    thread2 = Hello.myThread(2, url)
    thread3=Hello.myThread(3,url)
    # 开启线程
    thread1.start()
    thread2.start()
    thread3.start()
    thread1.join()
    thread2.join()
    thread3.join()
    res1 = thread1.get_result()#车道线
    res2=thread2.get_result()#车辆检测
    res3=thread3.get_result()#交通信号灯
    result=res1+res2+res3
    json_data = json.dumps(result)
    #print(res1)
    print(json_data)
    return json_data
@app.route('/load',methods=['POST','GET'])
def load():
    res=[]
    time=request.args.get('time')
    url="https://lindia.oss-cn-beijing.aliyuncs.com/drive4.mp4?x-oss-process=video/snapshot,t_%s,f_jpg,w_800,h_600"% time
    print(url)
    file = requests.get(url)
    img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    result,str=lane_finder.run(img)
    im2 = cv2.resize(result, (800, 500), )
    cv2.imwrite('result.png', im2)  # 写入图片
    #yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./result.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    res.append(raw_data)
    res.append(str)
    json_data = json.dumps(res)
    print(raw_data)
    return json_data

#处理信号灯
@app.route('/load1',methods=['POST','GET'])
def load1():
    time=request.args.get('time')
    url="https://lindia.oss-cn-beijing.aliyuncs.com/drive4.mp4?x-oss-process=video/snapshot,t_%s,f_jpg,w_800,h_600"% time
    print(url)
    file = requests.get(url)
    img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    result=lane_finder.run(img)#信号灯
    im2 = cv2.resize(result, (800, 500), )
    cv2.imwrite('result1.png', im2)  # 写入图片
    #yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./result.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    json_data = json.dumps(raw_data)
    print(raw_data)
    return json_data
#车辆检测
@app.route('/load2',methods=['POST','GET'])
def load2():
    res=[]
    time=request.args.get('time')
    url="https://lindia.oss-cn-beijing.aliyuncs.com/drive4.mp4?x-oss-process=video/snapshot,t_%s,f_jpg,w_800,h_600"% time
    print(url)
    file = requests.get(url)
    img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    result,str=car_finder.run(img)#车辆检测
    im2 = cv2.resize(result, (800, 500), )
    cv2.imwrite('result2.png', im2)  # 写入图片
    #yuan = cv2.imread('https://lindia.oss-cn-beijing.aliyuncs.com/project_video1.mp4?x-oss-process=video/snapshot,t_20000,f_jpg,w_800,h_600')
    # result=car_finder.run(yuan)
    # cv2.imwrite('a2.png', result)
    with open(r'./result2.png', 'rb') as f:  # 转化成二进制格式
        base64_bytes = base64.b64encode(f.read())  # 使用base64对数据进行加密
    base64_string = base64_bytes.decode(ENCODING)
    raw_data = base64_string
    res.append(raw_data)
    res.append(str)
    print(str)
    json_data = json.dumps(res)
    return json_data


# @app.route('/postdata', methods=['POST'])
# def postdata():
#     f = request.files['content']
#
#     print(f)
#     user_input = request.form.get("name")
#     basepath = os.path.dirname(__file__)  # 当前文件所在路径
#     src_imgname = str(uuid.uuid1()) + ".jpg"
#     upload_path = os.path.join(basepath, 'static/srcImg/')
#     if os.path.exists(upload_path)==False:
#         os.makedirs(upload_path)
#     f.save(upload_path + src_imgname)
#     im = cv2.imread(upload_path + src_imgname, 0)
#
#     save_path = os.path.join(basepath, 'static/resImg/')
#     if os.path.exists(save_path) == False:
#         os.makedirs(save_path)
#     save_imgname = str(uuid.uuid1()) + ".jpg"
#     cv2.imwrite(save_path + save_imgname, im)
#     resSets["value"] = 10
#     resSets["resurl"] = "http://127.0.0.1:5000" +'/static/resImg/' + save_imgname
#     return json.dumps(resSets, ensure_ascii=False)

if __name__ == '__main__':
    app.debug = True
    app.run(threaded=True)
