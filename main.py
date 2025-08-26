from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC

import glob

image_list = []  # Stores 1d array of images as list
labels = []  # Stores label of image as a number from 0 to 6


# For loops for all kinds of image types
s=0
for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/angry/*.jpg'):
    im = Image.open(filename)  # Opens image
    im = np.asarray(im)
    image_list.append(im.flatten())  # Turn image into 1d array
    labels.append(0)  # Append number to labels list
    s+=1
    if s > 50:
        break
s = 0
for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/disgust/*.jpg'):
    im = Image.open(filename)
    im = np.asarray(im)
    image_list.append(im.flatten())
    labels.append(1)
    s += 1
    if s > 50:
        break

s=0
for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/fear/*.jpg'):
    im = Image.open(filename)
    im = np.asarray(im)
    image_list.append(im.flatten())
    labels.append(2)
    s += 1
    if s > 50:
        break

s=0
for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/happy/*.jpg'):
    im = Image.open(filename)
    im = np.asarray(im)
    image_list.append(im.flatten())
    labels.append(3)
    s += 1
    if s > 50:
        break

s=0
for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/neutral/*.jpg'):
    im = Image.open(filename)
    im = np.asarray(im)
    image_list.append(im.flatten())
    labels.append(4)
    s += 1
    if s > 50:
        break

s=0
for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/sad/*.jpg'):
    im = Image.open(filename)
    im = np.asarray(im)
    image_list.append(im.flatten())
    labels.append(5)
    s += 1
    if s > 50:
        break

s=0
for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/surprise/*.jpg'):
    im = Image.open(filename)
    im = np.asarray(im)
    image_list.append(im.flatten())
    labels.append(6)
    s += 1
    if s > 50:
        break

image_list = np.asarray(image_list)
labels = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(image_list, labels)

new_train = X_train
y_new_train = y_train
s = 0

clf = MLPClassifier(learning_rate_init=0.0001, hidden_layer_sizes=(10,10), max_iter=2000)
clf.fit(X_train, y_train)
pred_train_classification = clf.predict(X_train)
label = [0]*len(X_train)
for i in range (len(X_train)):
    if pred_train_classification[i] != y_train[i]:
        new_train[s] = X_train[i]
        y_new_train[s] = y_train[i]
        s+=1
        label[i] = 1

clf1 = MLPClassifier(learning_rate_init=0.0001, hidden_layer_sizes=(100), max_iter=2000)
clf1.fit(new_train,y_new_train)

clf2 = LogisticRegression(class_weight='balanced')
clf2.fit(X_train, label)
print(clf2.predict(X_test))
print(label)
#clf = MLPClassifier(learning_rate_init=0.0001, hidden_layer_sizes=(100,100), max_iter=2000)
#clf = KNeighborsClassifier(n_neighbors=3)
#clf = GaussianProcessClassifier()
#clf = SVC(gamma=2, C=1)
#clf.fit(X_train, y_train)

predictions = clf.predict(X_train)
predictions1 = clf.predict(X_test)
print(accuracy_score(y_train, predictions))
print(accuracy_score(y_test, predictions1))
print(len(new_train))

pred2 = clf2.predict(X_test)
print(pred2)
pred = clf.predict(X_test)
pred1=clf1.predict(X_test)

s = 0
for i in range(len(y_test)):
    if pred2[i] == 1:
        prediction = pred1[i]
    else:
        prediction = pred[i]
    if prediction == y_test[i]:
        s += 1
print(s/len(y_test))