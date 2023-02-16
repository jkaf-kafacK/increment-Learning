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


#create keras sequence object
class GITW_Sequence(tf.keras.utils.Sequence):
    def __init__(self, folder, size, batch_size):
        self.folder = folder
        self.x = list(range(size))
        with(open("data/y_{}/y_{}".format(self.folder, self.folder),"rb")) as fp:
            self.y = pickle.load(fp)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = []
        for i in (self.x[idx * self.batch_size:(idx + 1) * self.batch_size]):
            with(open("data/x_{}/img_{}".format(self.folder, i),"rb")) as fp:
                x = pickle.load(fp)
                batch_x.append(x)
        batch_x = np.array(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = tf.convert_to_tensor(batch_x, dtype="float32")
        batch_y = tf.convert_to_tensor(batch_y, dtype="float32")
        
        return batch_x, batch_y