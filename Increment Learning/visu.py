import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pickle
import keras
import tensorflow as tf
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from codecarbon import EmissionsTracker
import time
from keras.callbacks import CSVLogger
import pandas as pd
import seaborn as sns

classes = ["Bowl","CanOfCocaCola", "Jam", "MilkBottle", "Mug", "OilBottle", "Rice", "Sugar", "VinegarBottle"]

def analyze_data():
    with(open("data/y_train_bboxes/y_train_bboxes","rb")) as fp:
        y_train_bboxes = pickle.load(fp)
    y_train_bboxes_effectif = {"Bowl":np.count_nonzero(y_train_bboxes == 0),
                    "CanOfCocaCola":np.count_nonzero(y_train_bboxes == 1),
                    "Jam":np.count_nonzero(y_train_bboxes == 2),
                    "MilkBottle":np.count_nonzero(y_train_bboxes == 3),
                    "Mug":np.count_nonzero(y_train_bboxes == 4),
                    "OilBottle":np.count_nonzero(y_train_bboxes == 5),
                    "Rice":np.count_nonzero(y_train_bboxes == 6),
                    "Sugar":np.count_nonzero(y_train_bboxes == 7),
                    "VinegarBottle":np.count_nonzero(y_train_bboxes == 8)}
    print(y_train_bboxes_effectif)
    sns.barplot(x=list(y_train_bboxes_effectif.keys()),y=list(y_train_bboxes_effectif.values()))

    with(open("data/y_train_saliency/y_train_saliency","rb")) as fp:
        y_train = pickle.load(fp)
    y_train_effectif = {"Bowl":np.count_nonzero(y_train == 0),
                    "CanOfCocaCola":np.count_nonzero(y_train == 1),
                    "Jam":np.count_nonzero(y_train == 2),
                    "MilkBottle":np.count_nonzero(y_train == 3),
                    "Mug":np.count_nonzero(y_train == 4),
                    "OilBottle":np.count_nonzero(y_train == 5),
                    "Rice":np.count_nonzero(y_train == 6),
                    "Sugar":np.count_nonzero(y_train == 7),
                    "VinegarBottle":np.count_nonzero(y_train == 8)}
    print(y_train_effectif)
    sns.barplot(x=list(y_train_effectif.keys()),y=list(y_train_effectif.values()))

    with(open("data/y_test/y_test","rb")) as fp:
        y_test = pickle.load(fp)
    y_test_effectif = {"Bowl":np.count_nonzero(y_test == 0),
                        "CanOfCocaCola":np.count_nonzero(y_test == 1),
                        "Jam":np.count_nonzero(y_test == 2),
                        "MilkBottle":np.count_nonzero(y_test == 3),
                        "Mug":np.count_nonzero(y_test == 4),
                        "OilBottle":np.count_nonzero(y_test == 5),
                        "Rice":np.count_nonzero(y_test == 6),
                        "Sugar":np.count_nonzero(y_test == 7),
                        "VinegarBottle":np.count_nonzero(y_test == 8)}
    print(y_test_effectif)
    sns.barplot(x=list(y_test_effectif.keys()),y=list(y_test_effectif.values()))

def plot_confusion_matrix(model):
    y_pred = model.predict(test)
    y_pred = np.argmax(y_pred, axis=1) 
    with(open("data/y_test/y_test","rb")) as fp:
        y_test = pickle.load(fp)
    cm=confusion_matrix(y_test,y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)

def plot_model(name):
    model = keras.models.load_model(name)
    plot_confusion_matrix(model)
    log_data = pd.read_csv('training_{}.log'.format(name), sep=',', engine='python')
    plt.plot(log_data['val_sparse_categorical_accuracy'])
    plt.plot(log_data['sparse_categorical_accuracy'])
    plt.plot(log_data['val_loss'])
    plt.plot(log_data['loss'])
    plt.legend(["Val acc", "Train acc", "Val loss", "Train loss"])
    print(log_data['val_sparse_categorical_accuracy'])
    print(log_data['sparse_categorical_accuracy'])