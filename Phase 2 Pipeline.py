#!/usr/bin/env python
# coding: utf-8

# ## Imort Useful Libraries

# In[1]:


import numpy as np
import cv2 as cv
import time
import os
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from moviepy.editor import *
from IPython.display import HTML
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load YOLO weights and cfg

# In[2]:


weights_path1 = 'yolov3.weights'
cfg_path1 = 'yolov3.cfg'

print("Loaded!")


# ## Read The Labels File
# 

# In[3]:


labels_path = 'coco.names.txt'
labels = open(labels_path).read().strip().split("\n")


# ## Load Nural Nestwork in CV2

# In[4]:


net = cv.dnn.readNetFromDarknet(cfg_path1,weights_path1)


# ### Get Layers Names

# In[5]:


names = net.getLayerNames()


# In[6]:


Layers_Names = [names[i - 1] for i in net.getUnconnectedOutLayers()]


# ### Load Test Image

# In[7]:


def load_img(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


# ### Run the inference on the test image

# In[8]:


def run_inference(img,Layers_Names):
    blob = cv.dnn.blobFromImage(img,1/255.0,(416,416),crop = False , swapRB = False)
    net.setInput(blob)

    Layers_Output = net.forward(Layers_Names)
    (H , W) = img.shape[:2]
    
    #return Layers_Output
    boxes = []
    confidences = []
    classIDs = []
    
    for outputs in Layers_Output:
        for detection in outputs:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
        
            if (confidence > 0.85):
                box = detection[:4] * np.array([W,H,W,H])
                bx , by, bw, bh = box.astype("int")
             
                x = int(bx - (bw / 2))
                y = int(by - (bh / 2))
            
                boxes.append([x,y,int(bw),int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv.dnn.NMSBoxes(boxes,confidences,0.8,0.8)
    return boxes, confidences, classIDs, idxs


# ### Plot The Bounding Boxes in the Image

# In[9]:


def plot_Bounding_Boxes(img, boxes, confidences, classIDs, idxs):
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x , y) = [boxes[i][0],boxes[i][1]]
            (w , h) = [boxes[i][2],boxes[i][3]]
    
            cv.rectangle(img,(x,y), (x+w,y+h), (0,0,255),2)
            cv.putText(img,"{}: {}".format(labels[classIDs[i]] ,confidences[i]) , (x,y-5) , cv.LINE_AA , 0.5 ,(255,255,255),2 )
    
    return img


# ## Cars Detection Pipeline

# In[10]:


def car_pipeline(img):
    #img = load_img(input_path)
    boxes, confidences, classIDs, idxs = run_inference(img,Layers_Names)
    out_img = plot_Bounding_Boxes(img, boxes, confidences, classIDs, idxs)
    #cv.imshow("Image",cv.cvtColor(img, cv.COLOR_BGR2RGB))
    #cv.waitKey(0)
    return out_img
    


# In[11]:


def Create_Image2(input_path,output_path):
    img = load_img(input_path)
    new_img = car_pipeline(img)
    new_img = Image.fromarray(new_img.astype(np.uint8), 'RGB')
    new_img.save(output_path)
    #cv.imshow("Image",cv.cvtColor(img, cv.COLOR_BGR2RGB))
    #cv.waitKey(0)


# In[13]:


test_img_path = "C:/Users/comp/Downloads/Compressed/Project_data/test_images/test3.jpg"
out_path = "C:/Users/comp/Downloads/Compressed/Project_data/test_images/test3new.jpg"
Create_Image2(test_img_path,out_path)


# In[14]:


def Create_Video2(input_path,output_path):
    video_input = VideoFileClip(input_path)
    processed_video = video_input.fl_image(car_pipeline)
    get_ipython().run_line_magic('time', 'processed_video.write_videofile(output_path, audio=False)')


# In[15]:


input_path = 'challenge_video_video_output.mp4'


# In[16]:


output_path = 'challenge_video_outputP2.mp4'


# In[17]:


Create_Video2(input_path,output_path)


# In[ ]:




