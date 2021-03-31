# Databricks notebook source
# MAGIC %pip install opencv-python
# MAGIC %pip install tqdm

# COMMAND ----------

# https://github.com/takyamamoto/Fixation-Densitymap/blob/master/Fixpos2Densemap.py

import cv2
import numpy as np
from tqdm import tqdm

def GaussianMask(sizex,sizey, sigma=33, center=None,fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0])==False and np.isnan(center[1])==False:            
            x0 = center[0]
            y0 = center[1]        
        else:
            return np.zeros((sizey,sizex))

    return fix*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def Fixpos2Densemap(fix_arr, width, height, imgfile, alpha=0.5, threshold=10):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap 
    """
    
    heatmap = np.zeros((H,W), np.float32)
    for n_subject in tqdm(range(fix_arr.shape[0])):
        heatmap += GaussianMask(W, H, 33, (fix_arr[n_subject,0],fix_arr[n_subject,1]),
                                fix_arr[n_subject,2])

    # Normalization
    heatmap = heatmap/np.amax(heatmap)
    heatmap = heatmap*255
    heatmap = heatmap.astype("uint8")
    
    if imgfile.any():
        # Resize heatmap to imgfile shape 
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create mask
        mask = np.where(heatmap<=threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
        marge = imgfile*mask + heatmap_color*(1-mask)
        marge = marge.astype("uint8")
        marge = cv2.addWeighted(imgfile, 1-alpha, marge,alpha,0)
        return marge

    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt

dataDir = "/dbfs/mnt/S3_rtl-databricks-datascience/datasets/EEG-data/"
#os.listdir(dataDir + 'Sensor Data')

d = pd.read_csv(dataDir + 'data@granularity/' + 'perFrame.csv')
d = d[d.SlideEvent != "StartSlide"]
d["frame"] = d.groupby(['patient','SourceStimuliName']).cumcount()+1

# COMMAND ----------

os.listdir(dataDir + 'videoland-trailers/')

# COMMAND ----------

f = 80
show = "Drag Race"
#Mocro Maffia


vidcap = cv2.VideoCapture(dataDir + 'videoland-trailers/' + show + '.mp4')
vidcap.set(1,f)
success,image = vidcap.read()

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#Show the image with matplotlib
# cv2.imshow('image',image) does not work in notebooks
#plt.show()

# COMMAND ----------

fix = d[(d["SourceStimuliName"] == show) & (d["frame"] == f)]
x = fix[["ET_CameraLeftX", "ET_CameraRightX"]].mean(axis = 1) * image.shape[1]
y = fix[["ET_CameraLeftY", "ET_CameraRightY"]].mean(axis = 1) * image.shape[0]

# only left eye (I tried to invert the y axis)
# x = fix["ET_CameraLeftX"] * image.shape[1]
# y = (1 - fix["ET_CameraLeftY"]) * image.shape[0]

# only right eye
# x = fix["ET_CameraRightX"] * image.shape[1]
# y = fix["ET_CameraRightY"] * image.shape[0]

fix_arr = np.column_stack((x, y, np.ones(d.patient.nunique())))

num_subjects = d.patient.nunique()
H, W, _ = image.shape

# Create heatmap
heatmap = Fixpos2Densemap(fix_arr, W, H, image, 0.7, 5)
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))