from PIL import Image
import matplotlib.image as pimg
import os

def Resize(Input_Path, Output_Path):
	
	for imagename in os.listdir(Input_Path):
		img = pimg.imread(Input_Path + imagename)
		im = Image.fromarray(img[:,:,0:3])
		im.resize((416,416))
		im.save(Output_Path + imagename)

Input_Path = "license-plate-dataset/Original_Images/train/"
Output_Path = "license-plate-dataset/images/train/"
Resize(Input_Path, Output_Path)

Input_Path = "license-plate-dataset/Original_Images/valid/"
Output_Path = "license-plate-dataset/images/valid/"
Resize(Input_Path, Output_Path)
