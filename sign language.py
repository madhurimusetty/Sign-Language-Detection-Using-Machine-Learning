#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install split-folders


# In[3]:


get_ipython().system('python.exe -m pip install --upgrade pip')


# In[1]:


import os 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import splitfolders

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D, BatchNormalization,Input,concatenate
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix


# In[5]:


# Path where our data is located
base_path = "C:\\Users\\madhu\\Downloads\\extracted_dataset\\asl_dataset\\"


# Dictionary to save our 36 classes
categories = {  0: "0",
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "a",
                11: "b",
                12: "c",
                13: "d",
                14: "e",
                15: "f",
                16: "g",
                17: "h",
                18: "i",
                19: "j",
                20: "k",
                21: "l",
                22: "m",
                23: "n",
                24: "o",
                25: "p",
                26: "q",
                27: "r",
                28: "s",
                29: "t",
                30: "u",
                31: "v",
                32: "w",
                33: "x",
                34: "y",
                35: "z",
            }

def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(
        lambda x: x[re.search("_", x).start() + 1 : re.search("_", x).start() + 2]
        + "/"
        + x
    )
    return df


# list conatining all the filenames in the dataset
filenames_list = []
# list to store the corresponding category, note that each folder of the dataset has one class of data
categories_list = []

for category in categories:
    filenames = os.listdir(base_path + categories[category])
    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df = pd.DataFrame({"filename": filenames_list, "category": categories_list})
df = add_class_name_prefix(df, "filename")

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)


# In[6]:


df


# In[7]:


print("number of elements = ", len(df))


# In[8]:


plt.figure(figsize=(40,40))

for i in range(100):
    path = base_path + df.filename[i]
    img = plt.imread(path)
    plt.subplot(10,10, i + 1)
    plt.imshow(img)
    plt.title(categories[df.category[i]],fontsize=35,fontstyle='italic')
    plt.axis("off")


# In[9]:


label,count = np.unique(df.category,return_counts=True)
uni = pd.DataFrame(data=count,index=categories.values(),columns=['Count'])

plt.figure(figsize=(14,4),dpi=200)
sns.barplot(data=uni,x=uni.index,y='Count',palette='icefire',width=0.4).set_title('Class distribution in Dataset',fontsize=15)
plt.show()


# In[13]:


import splitfolders

input_folder = r"C:\Users\madhu\Downloads\extracted_dataset\asl_dataset"
output_folder = r"C:\Users\madhu\Downloads\archive (2)"

splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=1333,
    ratio=(0.8, 0.1, 0.1)
)


# In[14]:


import shutil

shutil.rmtree(r"C:\Users\madhu\Downloads\archive (2)\train\asl_dataset")
shutil.rmtree(r"C:\Users\madhu\Downloads\archive (2)\val\asl_dataset")
shutil.rmtree(r"C:\Users\madhu\Downloads\archive (2)\test\asl_dataset")


# In[15]:


datagen = ImageDataGenerator(rescale= 1.0 / 255)


# In[16]:


train_path = r"C:\Users\madhu\Downloads\archive (2)\train"
val_path = r"C:\Users\madhu\Downloads\archive (2)\val"
test_path = r"C:\Users\madhu\Downloads\archive (2)\test"


batch = 32
image_size = 200
img_channel = 3
n_classes = 36


# In[17]:


train_data = datagen.flow_from_directory(directory= train_path, 
                                         target_size=(image_size,image_size), 
                                         batch_size = batch, 
                                         class_mode='categorical')

val_data = datagen.flow_from_directory(directory= val_path, 
                                       target_size=(image_size,image_size), 
                                       batch_size = batch, 
                                       class_mode='categorical',
                                       )

test_data = datagen.flow_from_directory(directory= test_path, 
                                         target_size=(image_size,image_size), 
                                         batch_size = batch, 
                                         class_mode='categorical',
                                         shuffle= False)


# In[18]:


model = Sequential()
# input layer
# Block 1
model.add(Conv2D(32,3,activation='relu',padding='same',input_shape = (image_size,image_size,img_channel)))
model.add(Conv2D(32,3,activation='relu',padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(64,3,activation='relu',padding='same'))
model.add(Conv2D(64,3,activation='relu',padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.3))

#Block 3
model.add(Conv2D(128,3,activation='relu',padding='same'))
model.add(Conv2D(128,3,activation='relu',padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.4))

# fully connected layer
model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(36, activation='softmax'))



model.summary()


# In[19]:


early_stoping = EarlyStopping(monitor='val_loss', 
                              min_delta=0.001,
                              patience= 5,
                              restore_best_weights= True, 
                              verbose = 0)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', 
                                         patience = 2, 
                                         factor=0.5 , 
                                         verbose = 1)


# In[20]:


model.compile(optimizer='adam', loss = 'categorical_crossentropy' , metrics=['accuracy'])


# In[21]:


asl_class = model.fit(train_data, 
                      validation_data= val_data, 
                      epochs=30, 
                      callbacks=[early_stoping,reduce_learning_rate],
                      verbose = 1)


# In[22]:


# Evaluvate for train generator
loss,acc = model.evaluate(train_data , verbose = 0)

print('The accuracy of the model for training data is:',acc*100)
print('The Loss of the model for training data is:',loss)

# Evaluvate for validation generator
loss,acc = model.evaluate(val_data, verbose = 0)

print('The accuracy of the model for validation data is:',acc*100)
print('The Loss of the model for validation data is:',loss)


# In[23]:


# plots for accuracy and Loss with epochs

error = pd.DataFrame(asl_class.history)

plt.figure(figsize=(18,5),dpi=200)
sns.set_style('darkgrid')

plt.subplot(121)
plt.title('Cross Entropy Loss',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.plot(error['loss'])
plt.plot(error['val_loss'])

plt.subplot(122)
plt.title('Classification Accuracy',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.plot(error['accuracy'])
plt.plot(error['val_accuracy'])

plt.show()


# In[24]:


# prediction
result = model.predict(test_data,verbose = 0)

y_pred = np.argmax(result, axis = 1)

y_true = test_data.labels

# Evaluvate
loss,acc = model.evaluate(test_data,verbose = 0)

print('The accuracy of the model for testing data is:',acc*100)
print('The Loss of the model for testing data is:',loss)


# In[25]:


p = y_pred
y = y_true
correct = np.nonzero(p==y)[0]
incorrect = np.nonzero(p!=y)[0]

print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])


# In[26]:


print(classification_report(y_true, y_pred,target_names= categories.values()))


# In[27]:


sns.set_style('ticks')

confusion_mtx = confusion_matrix(y_true,y_pred) 

f,ax = plt.subplots(figsize = (20,8),dpi=200)

sns.heatmap(confusion_mtx, annot=True, 
            linewidths=0.1, cmap = "gist_yarg_r", 
            linecolor="black", fmt='.0f', ax=ax, 
            cbar=False, xticklabels=categories.values(), 
            yticklabels=categories.values())

plt.xlabel("Predicted Label",fontdict={'color':'red','size':20})
plt.ylabel("True Label",fontdict={'color':'green','size':20})
plt.title("Confusion Matrix",fontdict={'color':'brown','size':25})

plt.show()


# In[30]:


# Import necessary libraries
import os
import re
import numpy as np
import pandas as pd
from keras.preprocessing import image  # Add this import
from keras.applications.vgg16 import preprocess_input  # Add this import
from keras.models import load_model

# Define global variables
image_size = 200
img_channel = 3
categories = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f", 16: "g", 17: "h", 18: "i", 19: "j",
    20: "k", 21: "l", 22: "m", 23: "n", 24: "o", 25: "p", 26: "q", 27: "r", 28: "s", 29: "t",
    30: "u", 31: "v", 32: "w", 33: "x", 34: "y", 35: "z"
}

# Function to preprocess a single image for prediction
def preprocess_image(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(image_size, image_size))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand the dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to be between 0 and 1
    img_array /= 255.0
    
    return img_array

# Function to predict the class of a single image
def predict_single_image(image_path, loaded_model):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Reshape the image to match the input shape of the model
    preprocessed_image = np.reshape(preprocessed_image, (1, image_size, image_size, img_channel))

    # Make predictions
    predictions = loaded_model.predict(preprocessed_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Map the class index to the actual class label using your 'categories' dictionary
    predicted_class = categories[predicted_class_index]

    return predicted_class

# Load the pre-trained model
loaded_model = model

# Example usage
image_path_to_predict = r"C:\Users\madhu\Downloads\archive (2)\test\j\hand5_j_bot_seg_1_cropped.jpeg"
predicted_class = predict_single_image(image_path_to_predict, loaded_model)

print(f"The predicted class for the image is: {predicted_class}")


# In[29]:


import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the pre-trained model
loaded_model = model

# Define global variables
image_size = 200
img_channel = 3
categories = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f", 16: "g", 17: "h", 18: "i", 19: "j",
    20: "k", 21: "l", 22: "m", 23: "n", 24: "o", 25: "p", 26: "q", 27: "r", 28: "s", 29: "t",
    30: "u", 31: "v", 32: "w", 33: "x", 34: "y", 35: "z"
}

# Function to preprocess a single image for prediction
def preprocess_image(img):
    # Resize the image to match the model's expected input shape
    img = cv2.resize(img, (image_size, image_size))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize pixel values to be between 0 and 1
    img_array /= 255.0

    return img_array

# Function to predict the class of a single image
def predict_single_image(img, loaded_model):
    # Preprocess the image
    preprocessed_image = preprocess_image(img)

    # Make predictions
    predictions = loaded_model.predict(preprocessed_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Map the class index to the actual class label using your 'categories' dictionary
    predicted_class = categories[predicted_class_index]

    return predicted_class

# Open a connection to the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Perform prediction on the frame
    predicted_class = predict_single_image(frame, loaded_model)

    # Display the predicted class on the frame
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the processed frame with prediction
    cv2.imshow('Processed Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:


q

