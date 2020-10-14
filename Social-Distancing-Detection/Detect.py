import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
from matplotlib.patches import Rectangle
import tensorflow as tf
from ProcessingFunctions import *
import cv2

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

YOLOModel = tf.keras.models.load_model('Model/YoloV3_Model.h5')
IOU_Threshold = 0.5
NMS_Threshold = 0.5
Anchors = [[116,90, 156,198, 373,326], [30,61,62,45, 59,119], [10,13, 16,30, 33,23]]
Class_Threshold = 0.6
Labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def PerformPrediction(ImagePath, Input_W = 416,Input_H = 416):
	img = pimg.imread(ImagePath)
	Image_W, Image_H = img.shape[0], img.shape[1]
	I = (cv2.resize(img,(Input_W, Input_H)).astype(float))/255.0
	
	Prediction = YOLOModel.predict(np.expand_dims(I,axis=0))
	
	Boxes = []
	for i in range(len(Prediction)):
		Boxes += DecodeNetworkOutput(Prediction[i][0], Anchors[i], Class_Threshold, NMS_Threshold, Input_H, Input_W)
		
	CorrectBoxes(Boxes,Image_W, Image_H, Input_W,Input_H)
	NonMaxSupression(Boxes, NMS_Threshold)
	
	BoundingBoxes, BoxLabels, BoxScores = getBoxes(Boxes, Labels, Class_Threshold)
	
	for i in range(len(BoundingBoxes)):
	    print(BoxLabels[i], BoxScores[i])
	
	
PerformPrediction('People.jpg')
	
