import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import seaborn as sns
import os
from keras.applications.vgg19 import VGG19
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

print(os.listdir("D:\\malignant vs benign\\"))

SIZE = 256

images = []
labels = []
for directory_path in glob.glob("D:/malignant vs benign/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path,"*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
        labels.append(label)
images = np.array(images)
labels = np.array(labels)

le = preprocessing.LabelEncoder()
le.fit(labels)
labels_encoded = le.transform(labels)

x_train, x_test, y_train, y_test = train_test_split(images, labels_encoded, train_size=0.5)

print("X-Train",x_train.shape)
print("Y-Train",y_train.shape)
print("X-Test",x_test.shape)
print("Y-Test",y_test.shape)

# Normalize pixel values between 0 to 1
x_train, x_test = x_train/255.0, x_test/255.0

# One hot encode y values for neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Load model without classifier layers
VGG_model = VGG19(weights="imagenet", include_top=False, input_shape = (SIZE,SIZE,3))
VGG_model.summary()

for layer in VGG_model.layers:
    layer.trainable = False
VGG_model.summary()

feature_extractor = VGG_model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF = features

RF_model = RandomForestClassifier(n_estimators=128, random_state=42)

RF_model.fit(X_for_RF, y_train)

X_test_features = VGG_model.predict(x_test)

X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

prediction_RF = RF_model.predict(X_test_features)
#prediction_RF = le.inverse_transform(prediction_RF)
display(prediction_RF)

print("Accuracy : {}".format(metrics.accuracy_score(y_test, prediction_RF)))

cm = confusion_matrix(y_test, prediction_RF)
sns.heatmap(cm, annot=True)

n = np.random.randint(0, x_test.shape[0])
img = x_test[n]
#display(img)
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_features = VGG_model.predict(input_img)
input_img_features = input_img_features.reshape(input_img_features.shape[0], -1)
prediction_RF = RF_model.predict(input_img_features)[0]
prediction_RF = le.inverse_transform([prediction_RF])
print("The prediction for the image is \t {}".format(prediction_RF[0].title()))
print("The actual label for the image is \t {}".format(le.inverse_transform([labels_encoded[n]])[0].title()))
