"""
Created on Tue May 11 00:09:38 2022

Using VGG16 for feature extraction and Random Forest for classification

@author: TAN FEI ZHI B19EC0041
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16


# Read input images and assign labels based on folder names
print(os.listdir("Image/"))


#Resize images
SIZE = 256  

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 


##################Training##################

for directory_path in glob.glob("Image/Dataset/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


##################Testing/Validation##################


test_images = []
test_labels = [] 
for directory_path in glob.glob("Image/Dataset/validationF/*"):
    footprint_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(footprint_label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

###################################################################

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#############################
#Load model without classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
#For loop to each VGG model layers
#Look at params after training
for layer in VGG_model.layers:
	layer.trainable = False
    
#Trainable parameters will be 0 since it is false
VGG_model.summary()  

#############################

#Now, let us use features from convolutional network for RF
feature_extractor=VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF = features #This is our X input to RF


############Classification Part-Random Forest#############

#RANDOM FOREST
import time
from sklearn.ensemble import RandomForestClassifier

#X-axis = estimators(The number of trees you want to build before taking the maximum voting or averages of predictions. )
#higher estimators, slow but higher performance
#Random state: to set the seed for the random generator so that we can ensure that the results that we get can be reproduced.
RF_model = RandomForestClassifier(n_estimators = 30, random_state = 12)


start = time.time()

# Train the model on training data
RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding

stop = time.time()

#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

#Now predict using the trained RF model. 

prediction_RF = RF_model.predict(X_test_features)

#Inverse le transform to get original label back. (decode)
prediction_RF = le.inverse_transform(prediction_RF)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
print("The accuracy rate is: ", (metrics.accuracy_score(test_labels, prediction_RF))*100,"%")


#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)
print(cm)

#heatmap-Plot rectangular data as a color-encoded matrix
sns.heatmap(cm, annot=True)

#choose randomly
#n=np.random.randint(0, x_test.shape[0])
#specify on which image to be selected 
#belongs to baby 7
n=6
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_RF = RF_model.predict(input_img_features)[0] 
prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])
print("The total training time used:", {stop - start},"seconds")



