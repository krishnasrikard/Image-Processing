"""
Solving using Denoising AutoEncoder
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import seaborn as sns
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import argparse
import cv2
from tensorflow.keras.layers import BatchNormalization,Conv2D,Conv2DTranspose,LeakyReLU,Activation,Flatten,Dense,Reshape,Input
from tensorflow.keras.models import Model,load_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Creating Model
class DAE():
	@staticmethod
	def buildmodel(height,width,depth,filters=(64,128),latentDim=128):
		
		# Encoder Architecture
		inputs = Input(shape=(height,width,depth))
		x = inputs
		for f in filters:
			x = Conv2D(f, (3,3), strides=2, padding='same')(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=-1)(x)
			
		volumeSize = tf.keras.backend.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim)(x)
		
		# Encoder Model
		Encoder = Model(inputs,latent,name="Encoder")
		
		
		# Decoder Architecture
		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)
		for f in filters[::-1]:
			x = Conv2DTranspose(f, (3,3), strides=2, padding='same')(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=-1)(x)
		x = Conv2DTranspose(depth, (3,3), padding='same')(x)
		output = Activation("sigmoid")(x)
		
		# Decoder Model
		Decoder = Model(latentInputs,output,name="Decoder")
		
		# Encoder and Decoder --> AutoEncoder
		AutoEncoder = Model(inputs,Decoder(Encoder(inputs)),name="AutoEncoder")
		
		return (Encoder,Decoder,AutoEncoder)
		
dae = DAE()

# Importing Data
def ReadTrainDataset(TrainImages):
	X_Train = []
	Y_Train = []
	
	for f in TrainImages:
		XPath = 'Dataset/train/' + f
		YPath = 'Dataset/train_cleaned/' + f
		X = Image.open(XPath)
		X = X.resize((256,256))
		X = np.array(X)
		
		Y = Image.open(YPath)
		Y = Y.resize((256,256))
		Y = np.array(Y)
		
		X_Train.append(X)
		Y_Train.append(Y)
		
	return np.array(X_Train), np.array(Y_Train)

TrainImages = os.listdir('Dataset/train')
X_Train, Y_Train = ReadTrainDataset(TrainImages)
print ("Imported Training Dataset")

X_Train = np.expand_dims(X_Train,axis=-1)
Y_Train = np.expand_dims(Y_Train,axis=-1)

X_Train = X_Train/255.0
Y_Train = Y_Train/255.0

print (X_Train.shape)

# Creating Model
Encoder,Decoder,AutoEncoder = dae.buildmodel(256,256,1)
print (AutoEncoder.summary())
tf.keras.utils.plot_model(Encoder, to_file='Images/Encoder.png', show_shapes=True, show_layer_names=True)
tf.keras.utils.plot_model(Decoder, to_file='Images/Decoder.png', show_shapes=True, show_layer_names=True)
tf.keras.utils.plot_model(AutoEncoder, to_file='Images/AutoEncoder.png', show_shapes=True, show_layer_names=True)

# Training Model
AutoEncoder.compile(loss='mse', optimizer='adam')
Epochs=100
Batch_Size=1

Train_History = AutoEncoder.fit(X_Train,Y_Train,validation_split=0.1,epochs=Epochs,batch_size=Batch_Size)
AutoEncoder.save("Models/DAE_Model.h5")

# Plotting Train Results
N = np.arange(Epochs)
plt.style.use("seaborn-poster")
plt.figure()
plt.plot(N, Train_History.history["loss"], label="Training Loss")
plt.plot(N, Train_History.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig("Images/Loss.png")
