import cv2
import threading

# video_process
import argparse
import numpy as np
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from PIL import Image, ImageDraw, ImageFont
import time
import pygame
import multiprocessing

Net = cv2.dnn.readNet('models/face_mask_detection.caffemodel', 'models/face_mask_detection.prototxt')

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
id2chiclass = {0: '佩戴口罩', 1: '没有佩戴口罩'}
colors = ((0, 255, 0), (255, 0 , 0))
flag_p = 1

flag = False

# 加载中文
def puttext_chinese(img, text, point, color):
    pilimg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilimg)
    fontsize = int(min(img.shape[:2])*0.04)
    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    y = point[1]-font.getsize(text)[1]
    if y <= font.getsize(text)[1]:
        y = point[1]+font.getsize(text)[1]
    draw.text((point[0], y), text, color, font=font)
    img = np.asarray(pilimg)
    return img

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    # print(net.getUnconnectedOutLayers())
    # for i in net.getUnconnectedOutLayers():
    #     layersNames[i[0] - 1] = i
    #     print(i)
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def inference(image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), draw_result=True, chinese=True):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=target_shape)
    net=Net
    net.setInput(blob)
    #
    y_bboxes_output, y_cls_output = net.forward(getOutputsNames(net))
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    # keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
    tl = round(0.002 * (height + width) * 0.5) + 1  # 厚度
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.

        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        if draw_result:
            '''
            绘制矩形
            image : 绘制的图片
            (xmin,ymin) : 绘制矩形的左上角
            (xmax,ymax) : 绘制矩形的右下角
            color : 绘制图片饿颜色
            thickness : 绘制矩形的宽度 -1 表示填满
            '''
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)

            if chinese:
                image = puttext_chinese(image, id2chiclass[class_id], (xmin, ymin), colors[class_id])  ###puttext_chinese
            else:
                cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])
    return image


    def run(self):
        while self.isRunning:
            # 这里cv2.VideoCapture() read方法表示读取每一帧    ret 表示读取帧的正确性     frame表示读取每一帧的图片 是一个三维矩阵
            ret, frame = self.cap.read()
            print(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 加载模型返回结果
            frame = inference(frame, target_shape=(260, 260), conf_thresh=0.5)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()

class VideoCamera(object):
    def __init__(self, _url,_flag):
        if (_url == '0'):
            self.url = int(_url)
        else:
            self.url = _url
        self.flag = True
        self.cap = cv2.VideoCapture(0)
        self.out = None

    #     # 打开摄像头， 0代表笔记本内置摄像头
    #
    #     # self.cap = cv2.VideoCapture("rtsp://10.3.120.224:8554/live")
    #     # self.cap = cv2.VideoCapture("http://10.3.120.224:8081/video")
    #     # if(flag == True):
    #     self.cap = cv2.VideoCapture("http://192.168.43.1:8081/video")
        # else:
        # self.cap = cv2.VideoCapture(0)

        # 视频录制线程
        # self.recordingThread = None

    # 退出程序释放摄像头
    def __del__(self):
        self.cap.release()

    def set(self,_flag,_url):
        self.flag = _flag
        if( _url == '0' ):
            self.url = 0
        else:
            self.url = "http://"+_url+"/video"

    def get_frame(self):
        if(not self.flag):
            self.cap = cv2.VideoCapture(self.url)
            self.falg = True
        ret, frame = self.cap.read()
        print(frame)
        # 转化图片为 rgb格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 加载模型
        frame = inference(frame, target_shape=(260, 260), conf_thresh=0.5)
        # 转化图片为bgr格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if ret:
            # 图片压缩  tips: 这里传入的必须是bgr格式的
            ret, jpeg = cv2.imencode('.jpg', frame)
            if self.out != None:
                self.out.release()
                self.out = None
            return jpeg.tobytes()

        else:
            return None
