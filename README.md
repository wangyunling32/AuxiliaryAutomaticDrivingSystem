# AuxiliaryAutomaticDrivingSystem
“领航”辅助自动驾驶系统

技术栈：Python+OpenCV+微信小程序 
项目描述：基于计算机视觉的辅助自动驾驶应用，能满足在驾驶员驾驶过程中 对环境中的车道线、交通信号灯、车辆进行检测和识别，并把检测结果反馈给 驾驶员，提醒驾驶员安全驾驶

成果展示

首页(驾驶员通过点击首页“现场应用”按钮，进入实时场景模块，系统就会自动捕获现场路况，并对图像进行采集;点击主页面的“现场模拟”按钮，进入模拟模块)：									

<img src="https://github.com/wangyunling32/AuxiliaryAutomaticDrivingSystem/blob/master/img-folder/homePage.png" width="300" height="500"/>

摄像头拍摄画面(通过实时捕捉小程序界面显示的摄像头拍摄画面完成图像采集;对采集的图像进行保存并存储到服务器，覆盖之前保存的图像)：

<img src="https://github.com/wangyunling32/AuxiliaryAutomaticDrivingSystem/blob/master/img-folder/camera.png" width="300" height="300"/>

识别结果(后端通过使用opencv对图像的处理办法对存储的图像进行预处理;通过使用相关技术编写车道线检测、车辆检测及交通信号灯识别等算法，对预处理的图像进行分析，并保存分析结果返回前端;前端对分析结果进行显示并用语音和文字提示驾驶员)：

<img src="https://github.com/wangyunling32/AuxiliaryAutomaticDrivingSystem/blob/master/img-folder/result.png" width="700" height="400"/>

该系统的核心是对图像的处理和分析，为了响应前端实时发送的路况分析结果请求，后端就必须快速的对保存的图像进行处理和分析，所以在处理图像时采用多进程同时执行的方法，减少图像处理和分析的时间，迅速的将分析结果反馈给前端，尽量与路况实时的内容保持同步。
