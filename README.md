# SA-C-GENDER-CLASSIFIER
# Algorithm
1.Import the necessary packages such as tensorflow, tensorflow hub, pandaa,matplotlib, and splitfloders.

2.Create your dataset of male and female images.

3.Using tensorflow.keras.preprocessing.image generate same image data for all the train and test images.

4.Using tensorflow.keras.preprocessing preprocess the images into numbers for gender prediction.

5.From tensorflow hub use B0 Feature vectors of EfficientNet models trained on Imagenet.

6.Use fit() to train the model.

7.Predict the gender of the image.

8.Create a loss and accuracy graph for understanding the model learning rate.

## Program:
```
/*
Program to implement 
Developed by   :ARR.Vijayaragavan
RegisterNumber :212220230059
*/
```
```python
import splitfolders  # or import split_folders
splitfolders.ratio("Male and Female face dataset", output="output", seed=1337, ratio=(.9, .1), group_prefix=None) # default values

import matplotlib.pyplot as plt
import matplotlib.image as mping
img = mping.imread('iu.jpg')
plt.imshow(img)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory("output/train/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")
test = train_datagen.flow_from_directory("output/val/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")

from tensorflow.keras.preprocessing import image

test_image = image.load_img('iu.jpg', target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = tf.expand_dims(test_image,axis=0)
test_image = test_image/255.
test_image.shape

import tensorflow_hub as hub
m = tf.keras.Sequential([
hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"),
tf.keras.layers.Dense(2, activation='softmax')])

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)
m.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])

history = m.fit(train,epochs=2,steps_per_epoch=len(train),validation_data=test,validation_steps=len(test))

classes=train.class_indices
classes=list(classes.keys())

m.predict(test_image)

classes[tf.argmax(m.predict(test_image),axis=1).numpy()[0]]

import pandas as pd
pd.DataFrame(history.history).plot()
```

## OUTPUT:
/*
![174449843-13e22959-b97d-4b12-8b20-820f972ab5be](https://user-images.githubusercontent.com/75235488/174487748-0814dce9-5052-40d5-ab4b-36701fc4a060.PNG)
![174449846-e30391c6-1d32-4a06-b380-298cfb3b15a2](https://user-images.githubusercontent.com/75235488/174487764-f4357ca4-194e-42b2-be1b-145b114bbc5c.PNG)

![174449861-1c4ff489-b526-4ae7-a3d3-7afb75c2d1e9](https://user-images.githubusercontent.com/75235488/174487801-2c671f82-b71c-4111-90f1-a8e89eab6f11.PNG)
![174449865-974daf3e-8163-4da2-9fcf-9057ab9e46b1](https://user-images.githubusercontent.com/75235488/174487809-e6113647-1a82-4e90-9146-5651c7779933.PNG)


2. DEMO VIDEO YOUTUBE LINK:

*/
https://youtu.be/kVXaKbdFbsw

### RESULT:

The B0 Efficientnet NN model has been created for gender classification and has sucessfully predicted gender of the input images.
