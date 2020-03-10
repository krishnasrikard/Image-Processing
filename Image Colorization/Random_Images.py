import numpy as np
import os

path = os.getcwd()
path = path + '/Images'

os.chdir(path)
filenames = os.listdir(path)
i = 0
j = 0
for f in filenames:
	if i%9 == 0:
		a = "Image"+str(j)+".jpg"
		os.rename(f,a)
		j = j+1
		i = i+1
	else:
		os.remove(f)
		i = i+1
