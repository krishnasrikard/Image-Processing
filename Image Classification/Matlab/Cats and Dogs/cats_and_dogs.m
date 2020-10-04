%       Transfer Learning

%{
Transfer Learning is the process of modifing a existing Neural Network convinent to our application.
Transfer learning is an efficient solution for many problems. Training requires some data and computer time, but much less than training from scratch, and the result is a network suited to your specific problem.
%}
    
%       Typical workflow for transfer learning

%{
To perform transfer learning, you need to create three components:
An array of layers representing the network architecture. For transfer learning, this is created by modifying a pre-existing network such as AlexNet.
Images with known labels to be used as training data. This is typically provided as a datastore.
A variable containing the options that control the behavior of the training algorithm.
These three components are provided as the inputs to the trainNetwork function which returns the trained network as output.
You should test the performance of the newly trained network. If it is not adequate, typically you should try adjusting some of the training options and retraining.
%}

%   Now we will be importing "AlexNet" with a convinent name(deepnet) for our usage. Layers function will display the layers in the network.(Remove " ; " to see the output)
deepnetwork = alexnet
layers = deepnetwork.Layers

%   Now we will view the 2 images of cats in train folder and print their size.
img1 = imread('cat.1.jpg');
size(img1)
imshow(img1)

img2 = imread('cat.2.jpg');
size(img2)
imshow(img2)

%   It can be observed that sizes of 2 images should be same as size of the input layer of "AlexNet". The below code gives the size of input layer.
Input_image_size = layers(1).InputSize

%   Pre-processing Images in a Datastore

%{
To classify images with a convolutional neural network, each image needs to have the size specified by the network's input layer. 
Images usually require simple preprocessing before they can be classified. You can preprocess your images individually, but it is common to perform the same preprocessing steps on the entire data set. It is more efficient to apply these steps to the entire datastore.
It can be time consuming to process each image individually. If you want to use an image datastore, you also need to save the processed images to a folder. For large data sets, saving a duplicate of your files can take up a lot of space.

It can be time consuming to process each image individually. If you want to use an image datastore, you also need to save the processed images to a folder. For large data sets, saving a duplicate of your files can take up a lot of space.
You can perform basic preprocessing with the augmentedImageDatastore function, which takes an image datastore and an image size as an input.
%}  

%   Creates a Datastore by not only taking images in the folder but also images in subfolders and labels them according to name of subfolder.
ds_pets = imageDatastore('Images','IncludeSubfolders',true,'LabelSource','foldernames')
%   Pet_Categories = ds_pets.Labels       ----  Prints each Category of Image

%   Entire data set is split into Training and Validation Data. Images in SubFolders will also be divided. "0.8" indicates the ratio of total belonging to TrainImgs.
[trainImgs,testImgs] = splitEachLabel(ds_pets,0.8,'randomized')

trainImgs = augmentedImageDatastore([227 227],trainImgs)
%trainImgs.ReadFcn = @(loc)imresize(imread(loc),[227 227]);


%   Actual Labels of Test Images
test_labels = testImgs.Labels


testImgs = augmentedImageDatastore([227 227],testImgs)
%testImgs.ReadFcn = @(loc)imresize(imread(loc),[227 227]);
%   Returns no.of different classes of Images
numClasses = numel(categories(ds_pets.Labels))

%   Now we will modify the AlexNet Layers to our convinence.

%   We now create fullyconnectedlayer having (no.of classes) neurons as output.
fclayer = fullyConnectedLayer(numClasses)

% 23rd layer of AlexNet  should replaced by our layer
layers(23) = fclayer
    
%   You can use the classificationLayer function to create a new output layer for an image classification network. You can create new layers and overwrite an existing layer with the new layer in a single command.
layers(end) = classificationLayer

%   Now set Training Options
options = trainingOptions('sgdm','InitialLearnRate',0.003,'MaxEpochs',10)

% Perform Training
[Petnet,info] = trainNetwork(trainImgs, layers, options);
plot(info.TrainingLoss)

%   Using classify function to predict the output on Test set
test_preds = classify(Petnet,testImgs)


%   Fraction of correct labels
numCorrect = nnz(test_labels == test_preds)
fracCorrect = numCorrect/numel(test_preds)

%   Displays confunsion chart
confusionchart(test_labels,test_preds)

pause;

% Predicting my profile Picture
img1 = imread('Naidu.jpg');
img1 = imresize(img,[227 227]);
imshow(img1)
prediction1 = classify(Petnet,img1)

img2 = imread('Me.jpg');
img2 = imresize(img,[227 227]);
imshow(img2)
prediction2 = classify(Petnet,img2)

img3 = imread('Naidu_Sleeping.jpg');
img3 = imresize(img,[227 227]);
imshow(img3)
prediction3 = classify(Petnet,img3)
