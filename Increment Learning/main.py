import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pickle
import keras
import tensorflow as tf
from utils import *
import math
from gitw_Sequence import *
from training import *

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

to_import = False
if (to_import == True):
    import_data()
    import_data_test()

gitw_sequence_train = GITW_Sequence("train_saliency", 10760, 16)
gitw_sequence_train_bboxes = GITW_Sequence("train_bboxes", 11594, 16)
test = GITW_Sequence("test", 7307, 16)
test2 = GITW_Sequence("test2", 5755, 16)
test2_mtd = GITW_Sequence("test2", 5755, 1)

to_train = False
if (to_train == True):
    #train_model(gitw_sequence_train_bboxes, test, "bboxes_v2B3")
    train_model(gitw_sequence_train, test, "saliency_v2B3_dropout", dropout=False)
else:
    model = keras.models.load_model('saliency_v2B3')
    #model.get_layer('efficientnetv2-b3').trainable=False
    #model_bboxes = keras.models.load_model('model_v2B3_bboxes')