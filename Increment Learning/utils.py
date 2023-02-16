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

'''
DEPRECATED VERSION: Missing some function to split test2 into five parts, and some files names are not good
'''

classes = ["Bowl","CanOfCocaCola", "Jam", "MilkBottle", "Mug", "OilBottle", "Rice", "Sugar", "VinegarBottle"]

def save_data(x_train_saliency, y_train_saliency, x_train_bboxes, y_train_bboxes):
    print("Start save")
    with(open("data/y_train_saliency/y_train_saliency", "wb" )) as fp:
        pickle.dump(y_train_saliency, fp)
    with(open("data/y_train_bboxes/y_train_bboxes", "wb" )) as fp:
        pickle.dump(y_train_bboxes, fp)
    for i in range(len(x_train_saliency)):
        with(open("data/x_train_saliency/img_{}".format(i), "wb")) as fp:
            pickle.dump(x_train_saliency[i], fp)
    for i in range(len(x_train_bboxes)):
        with(open("data/x_train_bboxes/img_{}".format(i), "wb")) as fp:
            pickle.dump(x_train_bboxes[i], fp)
    print("End save")

def save_data_test(x_test, y_test, num):
    print("Start save")
    with(open("data/y_test{}/y_test".format(num), "wb" )) as fp:
        pickle.dump(y_test, fp)
    for i in range(len(x_test)):
        with(open("data/x_test{}/img_{}".format(num, i), "wb" )) as fp:
            pickle.dump(x_test[i], fp)
    print("End save")

def import_data_test(num):
    x_test = []
    y_test = []
    
    for label in range(len(classes)):
        print("Starting class", classes[label], "import")
        folders = os.listdir("test{}/{}".format(num,classes[label]))
        #Create data foreach sample
        for folder in folders:
            saliency_frames = []
            path = "test{}/{}/{}".format(num, classes[label], folder)

            #saliency
            #retrieve frame to keep in fixation points
            pts_columns = ["frame", "is_present", "x", "y"]
            fixation_pts = pd.read_csv('{}/fixation_points.txt'.format(path), sep=" ", header=None, names=pts_columns, on_bad_lines='warn')
            #keep index of frames where object is present
            is_present_idx = fixation_pts.index[fixation_pts['is_present'] == 1].tolist()

            video = cv2.VideoCapture('{}/{}.mp4'.format(path,folder))
            success = 1
            count = 0

            while success:
                success, image = video.read()
                #SALIENCY : keep only frames where the object is present
                if (count in is_present_idx and success):
                    #format frame number to match files
                    str_idx = str(count)
                    while len(str_idx)<5:
                        str_idx = "0" + str_idx
                    #read frame saliency
                    img = cv2.imread('{}/SaliencyMaps/Saliency_{}.png'.format(path,str_idx),0)
                    #threshold between 50 and 255 (value to find)
                    th, img_2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
                    #get contours
                    contours, _ = cv2.findContours(img_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    #get bounding box
                    if (contours == ()): #If object is not present (error in fixations_points.txt)
                        continue
                    x,y,w,h = cv2.boundingRect(contours[0])
                    #crop frame to bounding box
                    cropped_img = image[int(y):int(y+h),int(x):int(x+w)]
                    x_test.append(np.array(cropped_img))
                    y_test.append(label)
                    
                count += 1
    print("End import")
    
    x_test_clean = []
    for img in x_test:
        res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        x_test_clean.append(res)
    x_test = np.array(x_test_clean)
    y_test = np.array(y_test)
    
    save_data_test(x_test, y_test, num)
    return x_test, y_test

def import_data():
    x_train_saliency = []
    y_train_saliency = []
    x_train_bboxes = []
    y_train_bboxes = []
    
    for label in range(len(classes)):
        print("Starting class", classes[label], "import")
        folders = os.listdir("train1/{}".format(classes[label]))
        #Create data foreach sample
        for folder in folders:
            saliency_frames = []
            bboxes_frames= []
            path = "train1/{}/{}".format(classes[label], folder)

            #saliency
            #retrieve frame to keep in fixation points
            pts_columns = ["frame", "is_present", "x", "y"]
            fixation_pts = pd.read_csv('{}/fixation_points.txt'.format(path), sep=" ", header=None, names=pts_columns, on_bad_lines='warn')
            #keep index of frames where object is present
            is_present_idx = fixation_pts.index[fixation_pts['is_present'] == 1].tolist()

            #bboxe
            bboxe_exist = os.path.exists('{}/{}_2_bboxes.txt'.format(path,folder))
            if (bboxe_exist == True):
                #Retrieve bounding boxes
                bboxe_columns = ["frame", "is_present", "x", "y", "width", "height"]
                bboxe = pd.read_csv('{}/{}_2_bboxes.txt'.format(path,folder), sep=" ", header=None, names=bboxe_columns, on_bad_lines='warn')
                #Filter frame where the object is not present
                bboxe = bboxe[bboxe['is_present']==1]

            video = cv2.VideoCapture('{}/{}.mp4'.format(path,folder))
            success = 1
            count = 0

            while success:
                success, image = video.read()
                #SALIENCY : keep only frames where the object is present
                if (count in is_present_idx and success):
                    #format frame number to match files
                    str_idx = str(count)
                    while len(str_idx)<5:
                        str_idx = "0" + str_idx
                    #read frame saliency
                    img = cv2.imread('{}/SaliencyMaps/Saliency_{}.png'.format(path,str_idx),0)
                    #threshold between 50 and 255 (value to find)
                    th, img_2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
                    #get contours
                    contours, _ = cv2.findContours(img_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    #get bounding box
                    if (contours == ()): #If object is not present (error in fixations_points.txt)
                        continue
                    x,y,w,h = cv2.boundingRect(contours[0])
                    #crop frame to bounding box
                    cropped_img = image[int(y):int(y+h),int(x):int(x+w)]
                    x_train_saliency.append(np.array(cropped_img))
                    y_train_saliency.append(label)

                #BBOXE : keep only frames where the object is present
                if (bboxe_exist == True):
                    if (count in bboxe['frame'].values and success):
                        #crop frame to bounding box
                        x = bboxe[bboxe['frame']==count].x.values[0]
                        y = bboxe[bboxe['frame']==count].y.values[0]
                        w = bboxe[bboxe['frame']==count].width.values[0]
                        h = bboxe[bboxe['frame']==count].height.values[0]
                        cropped_img = image[int(y):int(y+h),int(x):int(x+w)]
                        x_train_bboxes.append(np.array(cropped_img))
                        y_train_bboxes.append(label)
                count += 1
    print("End import")
    x_clean = []
    for img in x_train_saliency:
        res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        x_clean.append(res)
    x_train_saliency = np.array(x_clean)
    y_train_saliency = np.array(y_train_saliency)
    
    x_clean = []
    for img in x_train_bboxes:
        res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        x_clean.append(res)
    x_train_bboxes = np.array(x_clean)
    y_train_bboxes = np.array(y_train_bboxes)
    
    save_data(x_train_saliency, y_train_saliency, x_train_bboxes, y_train_bboxes)
    return x_train_saliency, y_train_saliency, x_train_bboxes, y_train_bboxes
