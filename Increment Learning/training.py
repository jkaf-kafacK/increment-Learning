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

"""
For our model, we use transfer learning from EfficientNetV2, version B3 : https://arxiv.org/abs/2104.00298
"""

def train_model(train, test, name, dataAugmentation = False, dropout = False):
    efficientNet = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(
    include_top=False,
    weights='imagenet',
    input_shape=(256,256,3),
    pooling=None,
    include_preprocessing=True
    )
    efficientNet.trainable = False
    
    inputs = keras.Input(shape=(256, 256, 3))
    x = efficientNet(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(32, activation ='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(16, activation ='relu')(x)
    outputs = keras.layers.Dense(9, activation ='softmax')(x)
    model = keras.Model(inputs, outputs, name=name)
    model.summary()
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    csv_logger = CSVLogger('training_{}.log'.format(name), separator=',', append=False)
    tracker = EmissionsTracker(log_level = "warning", project_name = name, tracking_mode = "process")
    tracker.start()
    
    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy", dtype=None)])
    
    history = model.fit(train, epochs=20, validation_data=test, callbacks=[callback, csv_logger])
    tracker.stop()
    model.save(name)

def classic_fine_tuning(train, test, name):
    model = keras.models.load_model('saliency_v2B3')
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    csv_logger = CSVLogger('training_{}.log'.format(name), separator=',', append=False)
    tracker = EmissionsTracker(log_level = "warning", project_name = name, tracking_mode = "process")
    
    tracker.start()
    
    model._name="model_v2B3_fine_tuned"
    model.get_layer('efficientnetv2-b3').trainable=True
    
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy", dtype=None)])
    
    history = model.fit(train, epochs=10, validation_data=test, callbacks=[callback, csv_logger])
    tracker.stop()
    
    model.save(name)

def incremental_fine_tuning(train, test, trainable, origin, name):
    model = keras.models.load_model(origin)
    
    model.get_layer('efficientnetv2-b3').trainable=trainable
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    csv_logger = CSVLogger('training_{}.log'.format(name), separator=',', append=False)
    tracker = EmissionsTracker(log_level = "warning", project_name = name, tracking_mode = "process")
    
    tracker.start()
    
    model._name=name
    
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy", dtype=None)])
    
    history = model.fit(train, epochs=25, validation_data=test, callbacks=[callback, csv_logger])
    tracker.stop()
    
    model.save(name)

def move_to_data(data, epsilon, name):
    model = keras.models.load_model('saliency_v2B3_fine_tuned')
    
    tracker = EmissionsTracker(log_level = "warning", project_name = name, tracking_mode = "process")
    tracker.start()
    
    model2 = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    model2.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy", dtype=None)])
    
    weights = model.get_layer("dense_1").get_weights()
    n = 5755
    for k in range(n):
        if (k%500 == 0):
            print(k, "/", n)
        img, _ = data.__getitem__(k)
        pred = model2.predict(img, verbose=0)
        pred = pred.reshape((-1))
        for i in range(len(pred)):
            for j in range(len(weights[0][i])):
                weights[0][i][j] = weights[0][i][j] + ((np.linalg.norm(weights[0][i][j]) * (pred[i]/np.linalg.norm(pred[i]))) - weights[0][i][j])*epsilon
    
    model.get_layer("dense_1").set_weights(weights)
    
    tracker.stop()
    model.save(name)
    print("end")