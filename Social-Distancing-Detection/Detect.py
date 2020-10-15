import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
from matplotlib.patches import Rectangle
import tensorflow as tf
from ProcessingFunctions import *
import cv2

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

YOLOModel = tf.keras.models.load_model('Model/yolov3-spp.h5')
IOU_Threshold = 0.5
NMS_Threshold = 0.5
Anchors = [[116,90, 156,198, 373,326], [30,61,62,45, 59,119], [10,13, 16,30, 33,23]]
Class_Threshold = 0.6
Labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def Violation(Boxes,H,W):
	Centers = []
	Area = []
	
	for box in Boxes:
		x = box.xmin + box.xmax
		y = box.ymin + box.ymax
		Centers.append([x,y])
		Area.append(abs(box.xmin - box.xmax) * abs(box.ymin - box.ymax))
		
	Mask = np.zeros((len(Centers),1))
	Centers = np.array(Centers)
	
	for i in range(len(Centers)):
		for j in range(len(Centers)):
			print (i,j,np.linalg.norm(Centers[i]-Centers[j]),(abs(Area[i] - Area[j])/(H*W/1500)))
			if (i!=j):
				if ((abs(Area[i] - Area[j])/(H*W/1500)) < 300):
					if (np.linalg.norm(Centers[i]-Centers[j]) < 280):
						Mask[i] = 1
						Mask[j] = 1				
	return Mask
	
def DetectViolation(ImagePath, BoundingBoxes,BoxLabels):
	PeopleBoxes = []
	
	for i in range(len(BoxLabels)):
		if BoxLabels[i] == "person":
			PeopleBoxes.append(BoundingBoxes[i])
	
	img = pimg.imread(ImagePath)
	W,H = img.shape[0],img.shape[1]
	
	Mask = Violation(PeopleBoxes,H,W)
	
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in range(Mask.shape[0]):
		box = PeopleBoxes[i]
		y1, x1, y2, x2 = max(box.ymin,0), max(box.xmin,0), max(box.ymax,0), max(box.xmax,0)
		width, height = x2 - x1, y2 - y1
		
		if Mask[i] == 1:
			rect = Rectangle((x1, y1), width, height, fill=False, color='red', angle=0, lw = 1.5)
			label = "%s" % (i)
			ax.add_patch(rect)
			ax.text(x1,y1,label,color='red',fontsize=14)
		else:
			rect = Rectangle((x1, y1), width, height, fill=False, color='green', angle=0, lw = 1.5)
			label = "%s" % (i)
			ax.add_patch(rect)
			ax.text(x1,y1,label,color='green',fontsize=14)

	ax.imshow(img)
	plt.show()
	
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
	
	return BoundingBoxes, BoxLabels
	
	
BoundingBoxes, BoxLabels, = PerformPrediction('People-1.jpg')
DetectViolation('People-1.jpg',BoundingBoxes, BoxLabels)

BoundingBoxes, BoxLabels, = PerformPrediction('People-2.jpg')
DetectViolation('People-2.jpg',BoundingBoxes, BoxLabels)
	
