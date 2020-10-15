import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
from matplotlib.patches import Rectangle
import struct
import cv2
import random

def IntervalOverlap(A, B):
	# Finds Intersection Dimension
    x1, x2 = A
    x3, x4 = B

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
   
def Sigmoid(x):
	# Computes Sigmoid Function
    return 1. / (1. + np.exp(-x))


def IOU(box1, box2):
	# Intersection over Union
    W_intersection = IntervalOverlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    H_intersection = IntervalOverlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    IntersectionArea = W_intersection * H_intersection

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    UnionArea = w1*h1 + w2*h2 - IntersectionArea
    
    return float(IntersectionArea) / UnionArea

def PreprocessInput(image, Input_H, Input_W):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(Input_W)/new_w) < (float(Input_H)/new_h):
        new_h = (new_h * Input_W)/new_w
        new_w = Input_W
    else:
        new_w = (new_w * Input_H)/new_h
        new_h = Input_H

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((Input_H, Input_W, 3)) * 0.5
    new_image[int((Input_H-new_h)//2):int((Input_H+new_h)//2), int((Input_W-new_w)//2):int((Input_W+new_w)//2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def DecodeNetworkOutput(NetworkOutput, Anchors, Class_Threshold, NMS_Threshold, Input_H, Input_W):
    grid_h, grid_w = NetworkOutput.shape[:2]
    nb_box = 3
    NetworkOutput = NetworkOutput.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = NetworkOutput.shape[-1] - 5

    boxes = []

    NetworkOutput[..., :2]  = Sigmoid(NetworkOutput[..., :2])
    NetworkOutput[..., 4:]  = Sigmoid(NetworkOutput[..., 4:])
    NetworkOutput[..., 5:]  = NetworkOutput[..., 4][..., np.newaxis] * NetworkOutput[..., 5:]
    NetworkOutput[..., 5:] *= NetworkOutput[..., 5:] > Class_Threshold

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            objectness = NetworkOutput[int(row)][int(col)][b][4]            
            if(objectness.all() <= Class_Threshold): continue
            
            x, y, w, h = NetworkOutput[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # Center Position, unit: Image Width
            y = (row + y) / grid_h # Center Position, unit: Image Height
            w = Anchors[2 * b + 0] * np.exp(w) / Input_W # unit: Image Width
            h = Anchors[2 * b + 1] * np.exp(h) / Input_H # unit: Image Height  
            
            # last elements are class probabilities
            classes = NetworkOutput[int(row)][col][b][5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes

def CorrectBoxes(boxes, Image_H, Image_W, Input_H, Input_W):
	# Reseting Dimensions for all Boxes
	# Correction from Original Code
    if (float(Input_W)/Image_W) < (float(Input_H)/Image_H):
        new_w = Input_W
        new_h = (Image_H*Input_W)/Image_W
    else:
        new_h = Input_W
        new_w = (Image_W*Input_H)/Image_H
        
    for i in range(len(boxes)):
        x_offset, x_scale = (Input_W - new_w)/2./Input_W, float(new_w)/Input_W
        y_offset, y_scale = (Input_H - new_h)/2./Input_H, float(new_h)/Input_H
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * Image_W)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * Image_W)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * Image_H)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * Image_H)
        
def NonMaxSupression(boxes, NMS_Threshold):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if IOU(boxes[index_i], boxes[index_j]) >= NMS_Threshold:
                    boxes[index_j].classes[c] = 0

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def getLabel(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label
    
    def getScore(self):
        if self.score == -1:
            self.score = self.classes[self.getLabel()]
        return self.score

def Draw_Boxes(ImagePath, v_boxes, v_labels, v_scores):
	# Plotting Results
	img = pimg.imread(ImagePath)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		y1, x1, y2, x2 = max(box.ymin,0), max(box.xmin,0), max(box.ymax,0), max(box.xmax,0)
		width, height = x2 - x1, y2 - y1
		
		rect = Rectangle((x1, y1), width, height, fill=False, color='red', angle=0, lw = 2)
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		ax.add_patch(rect)
		ax.text(x1,y1,label,color='red',fontsize=14)

	ax.imshow(img)
	plt.show()

def getBoxes(Boxes, Labels, Class_Threshold):
	BoundingBoxes, BoxLabels, BoxScores = list(), list(), list()
	for box in Boxes:
		for i in range(len(Labels)):
			if box.classes[i] > Class_Threshold:
				BoundingBoxes.append(box)
				BoxLabels.append(Labels[i])
				BoxScores.append(box.classes[i]*100)
	return BoundingBoxes, BoxLabels, BoxScores
