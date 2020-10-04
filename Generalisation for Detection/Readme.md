# Generalisation of Detection
Training Models for Task of Object Detection

### Prequisites:
- Images should be in the folder **Dataset/Original_Images**. The Training and Validation Images should be present in their corresponding folders.
- Annotations of Images should be **Dataset/annots** folder. Corresponding Annotations of Training Images and Validation Images should be in their respective folders similar to **Original_Images**.
- Decide the Object-Detection Model from the ones available **cfg** folder.

### Procedure:
1. Input Image Dimensions of YOLO Models is (416,416,3). So, resize all the images to that Dimensions by running the code **resize-images.py**.
2. Extract Annotations. Feel free to modify it depending on **.xml** files. Make sure the labels of Images are the following order **(Class, X-Center, Y-Center, Width, Height)**.
3. Labels should be saved in **Dataset/labels** directory with corresponding Image-Names. As usual make sure corresponding labels of Training Images and Validation Images should be in their respective folders.
4. For Training Model should be provided with details of Images abd Labels and should be saved in **data** folder. Run the code **data-specs.py** to perform the operation. Don't worry about not generating specifications regarding labels. Training Procedure automatically changes the keyword **images** in path to **labels** for Labels of Images.
5. Modify the data in files as follows:
  - **model.cfg** should contain data of cfg file of selected YOLO-Model.
  - In **model.data** No.of Classes should be changed accordingly. The remaining details correspond to filenames created previous by running **data-specs.py** and **model.names** which will be modified in next step.
  - **model.names** contains names of classes.

### Training the Model:
#### Start Training
```
python3 train.py --cfg model.cfg --data data/model.data --epochs EPOCHS --nosave
```
The above commands trains the model for **EPOCHS** epochs and saves the weights after Training as **last.pt** in **weights** folder.

#### Resume Training
```
python3 train.py --cfg model.cfg --data data/model.data --epochs EPOCHS --weights weights/last.pt
```
The above commands resumes training the model with initial weights as **last.pt** in **weights** folder and trains for **EPOCHS** epochs.

#### Convert Weights for TensorFlow
```
python3  -c "from models import *; convert('cfg/model.cfg', 'weights/WEIGHTSNAME')"
```
Run the above commands to convert Weights from PyTorch Compatability to TensorFlow Compatability. WEIGHTSNAME corresponds to name of weights file (Eg: best.pt, last.pt).

### Testing
```
python3 detect.py --cfg data/model.cfg --names data/model.names --source Dataset/Test_Images --output Dataset/Results --weights weights/best.pt
```
Run the above command to test the Model.

### Source
https://github.com/ultralytics/yolov3
