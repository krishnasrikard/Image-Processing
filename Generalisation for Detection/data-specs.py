import os

# Training List
List = os.listdir('Dataset/images/train')
filename = open('data/'+ 'train_images.txt', "w")

for f in List:
	filename.write('../Dataset/images/train/'+f)
	filename.write('\n')
filename.close()

# Validation List
List = os.listdir('Dataset/images/valid')
filename = open('data/'+ 'valid_images.txt', "w")

for f in List:
	filename.write('../Dataset/images/valid/'+f)
	filename.write('\n')
filename.close()
