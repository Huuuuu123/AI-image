import torch
import cv2
import numpy as np
from PIL import Image


# 这个函数接收一个图片文件路径作为输入，
# 如果图像大小大于4000像素，则将图像整体等比缩小5倍，
# 如果大小在2000到4000之间，则缩小2.5倍，最后返回缩放后的图像对象。
# 如果图像大小小于等于2000，则不进行缩放，直接返回原图像。
def resize_image(image):
    width, height = image.size
    max_size = max(width, height)
    if max_size > 4000:
        ratio = 1 / 5.0
    elif max_size > 2000:
        ratio = 1 / 2.5
    else:
        return image  # 图像尺寸小于等于2000时不缩放
    
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return image.resize((new_width, new_height))


"默认一张图只有一个脸"

## 为实战而生
def crop_face(image):
    face_cascade = cv2.CascadeClassifier('face.xml')
    
    # 定义要剪切的图片大小
    crop_size = 256
    image = resize_image(image)
    # 将图片转为 numpy 数组
    img = np.array(image)
    
    # 将图片转为 RGB 格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        
        

        
        # 使用 MTCNN 模型检测和标定人脸
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            raise Exception("Unable to detect any faces in the input image.")
        # 处理人脸
        if faces is not None:
            for i, box in enumerate(faces):
                # 提取人脸框的坐标
                x1, y1, x2, y2 = box.astype(int)
                if x2 <= 256 and y2 <= 256:
                    x2, y2 = x1 + x2, y1 + y2
                    # 判断人脸是否越界
                    if x1 < 0 or x2 > img.shape[1] or y1 < 0 or y2 > img.shape[0]:
                        raise Exception(f"人脸{i + 1}越界，已忽略")
                    
                    # 计算剪切位置
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    x1c, y1c = cx - crop_size // 2, cy - crop_size // 2
                    x2c, y2c = cx + crop_size // 2, cy + crop_size // 2
                    
                    # 判断剪切位置是否越界
                    if x1c < 0:
                        x1c, x2c = 0, crop_size
                    elif x2c > img.shape[1]:
                        x1c, x2c = img.shape[1] - crop_size, img.shape[1]
                    if y1c < 0:
                        y1c, y2c = 0, crop_size
                    elif y2c > img.shape[0]:
                        y1c, y2c = img.shape[0] - crop_size, img.shape[0]
                    
                    # 剪切图像
                    crop_img = img[y1c:y2c, x1c:x2c]
                    
                    # 将剪切后的图像转为 PIL 格式
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
                else:
                    crop_img = img[y1:y1 + y2, x1:x1 + x2]
                    crop_img = cv2.resize(crop_img, (256, 256))
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
                # 返回剪切后的图像
                return crop_img
        
    except Exception as e:
        crop_img  = cv2.resize(img, (256,256))
        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        return crop_img
## 为测试而生
# 没有异常处理，可以在统计的时候处理
def crop_face_test(image):
    cnt = 0
    failures = []
    
    face_cascade = cv2.CascadeClassifier('face.xml')
    
    # 定义要剪切的图片大小
    crop_size = 256
    image = resize_image(image)
    # 将图片转为 numpy 数组
    img = np.array(image)
    
    # 将图片转为 RGB 格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 使用 MTCNN 模型检测和标定人脸
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        raise Exception("Unable to detect any faces in the input image.")
    # 处理人脸
    if faces is not None:
        for i, box in enumerate(faces):
            # 提取人脸框的坐标
            x1, y1, x2, y2 = box.astype(int)
            if x2 <= 256 and y2 <= 256:
                x2, y2 = x1 + x2, y1 + y2
                # 判断人脸是否越界
                if x1 < 0 or x2 > img.shape[1] or y1 < 0 or y2 > img.shape[0]:
                    raise Exception(f"人脸{i + 1}越界，已忽略")
                
                # 计算剪切位置
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                x1c, y1c = cx - crop_size // 2, cy - crop_size // 2
                x2c, y2c = cx + crop_size // 2, cy + crop_size // 2
                
                # 判断剪切位置是否越界
                if x1c < 0:
                    x1c, x2c = 0, crop_size
                elif x2c > img.shape[1]:
                    x1c, x2c = img.shape[1] - crop_size, img.shape[1]
                if y1c < 0:
                    y1c, y2c = 0, crop_size
                elif y2c > img.shape[0]:
                    y1c, y2c = img.shape[0] - crop_size, img.shape[0]
                
                # 剪切图像
                crop_img = img[y1c:y2c, x1c:x2c]
                
                # 将剪切后的图像转为 PIL 格式
                crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            else:
                crop_img = img[y1:y1 + y2, x1:x1 + x2]
                crop_img = cv2.resize(crop_img, (256, 256))
                crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            # 返回剪切后的图像
            return crop_img
    
    else:
        raise Exception("未检测到人脸！")

#
if __name__ == "__main__":
    img = Image.open("data_0000/5867276.jpg")
    print(np.array(img))


