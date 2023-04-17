# import keras as K

# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def fbeta_score(y_true, y_pred, beta=1):
#     if beta < 0:
#         raise ValueError('The lowest choosable beta is zero (only precision).')

#     # If there are no true positives, fix the F score at 0 like sklearn.
#     if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
#         return 0

#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     bb = beta ** 2
#     fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
#     return fbeta_score

# def fmeasure(y_true, y_pred):
#     return fbeta_score(y_true, y_pred, beta=1)



import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
from tool import evaluation_tool as et

import matplotlib.pyplot as plt
from keras.metrics import accuracy
import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix

filePath = "./final/label_and_pred.txt"


with open(filePath,mode='r',encoding='utf-8') as fileObj:
    lines = fileObj.readlines()

lines.pop(0)
#解析文件
for idx in range(len(lines)):
    lines[idx] = lines[idx].strip('\n')
    lines[idx] = lines[idx].split('\t')
    lines[idx].pop(len(lines[idx])-1)

y_label = []

for idx in range(len(lines)):
    y_label.append(lines[idx].pop(0))

y_pred = lines

#y_label = [int(label) for label in y_label]
y_label = [int(label) for label in y_label]
for idx in range(len(y_pred)):
    y_pred[idx] = [float(elem) for elem in y_pred[idx]]

y_pred = [np.argmax(pred) for pred in y_pred]

model = tf.keras.models.load_model('./model/model_10_200_200_tf_0417.h5')

C2= confusion_matrix(y_label, y_pred, labels=[0,1,2,3,4,5,6,7,8,9])

accuList = et.evaluate_accuracy(C2)
callList = et.evaluate_callback(C2)

pass

