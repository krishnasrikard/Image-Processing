import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import xml.etree.ElementTree as ET
import os

def readXmlDataset(xmlfile):
	Corpus = ET.parse(xmlfile).getroot()
	Box = []

	XMin = []
	XMax = []
	YMin = []
	YMax = []

	for a in Corpus.iter('filename'):
		Image_Name = a.text
				
	for a in Corpus.iter('width'):
		width = float(a.text)

	for a in Corpus.iter('height'):
		height = float(a.text)
				
	for a in Corpus.iter('xmin'):
		XMin.append(float(a.text)/width)
	
	for a in Corpus.iter('ymin'):
		YMin.append(float(a.text)/height)
	
	for a in Corpus.iter('xmax'):
		XMax.append(float(a.text)/width)
	
	for a in Corpus.iter('ymax'):
		YMax.append(float(a.text)/height)
				
	XMin = np.array(XMin)
	XMax = np.array(XMax)
	YMin = np.array(YMin)
	YMax = np.array(YMax)
	
	X_Center = (XMin + XMax)/2
	Y_Center = (YMin + YMax)/2
	Width = XMax-XMin
	Height = YMax-YMin
	
	Box = np.transpose(np.reshape(np.append(np.append(X_Center,Y_Center),np.append(Width,Height)),(4,-1)))

	Class = np.zeros((Box.shape[0],1))
	Box = np.append(Class,Box,axis=1)
	Box = Box.flatten()
	Box = Box.tolist()
	
	return Image_Name,Box
	
def ExtractData(path,dataset):
	base_path = os.getcwd()
	os.chdir(base_path + path)
	
	newpath = os.getcwd()
	AnnotsList = os.listdir(newpath)
	
	for f in AnnotsList:
		ImageName,Box = readXmlDataset(f)
		s = ''
		for i in range(len(Box)):
			s = s + str(Box[i]) + ' '
			if i+1>=5  and (i+1)%5 ==0:
				s = s + '\n'
		
		if dataset == 'train':		
			filename = open('../../labels/train/' + ImageName[:-4] + '.txt', "w")
		else:
			filename = open('../../labels/valid/' + ImageName[:-4] + '.txt', "w")
			
		filename.write(s)
		filename.close()
	
	os.chdir(base_path)

ExtractData('/license-plate-dataset/annots/train','train')
print ('------------ Train Data Extraction Completed------------------')
ExtractData('/license-plate-dataset/annots/valid', 'valid')
print ('------------ Validation Data Extraction Completed------------------')

# Training List
List = os.listdir('license-plate-dataset/images/train')
filename = open('Model_Data/'+ 'train_images.txt', "w")

for f in List:
	filename.write('../license-plate-dataset/images/train/'+f)
	filename.write('\n')
filename.close()

# Validation List
List = os.listdir('license-plate-dataset/images/valid')
filename = open('Model_Data/'+ 'valid_images.txt', "w")

for f in List:
	filename.write('../license-plate-dataset/images/valid/'+f)
	filename.write('\n')
filename.close()


