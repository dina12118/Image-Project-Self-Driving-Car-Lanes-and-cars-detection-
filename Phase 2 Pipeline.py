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
%matplotlib inline

weights_path1 = "C:/Users/comp/Desktop/Project_Output/yolov3.weights"
cfg_path1 = "C:/Users/comp/Desktop/Project_Output/yolov3.cfg"

print("Loaded!")


labels_path = "C:/Users/comp/Desktop/Project_Output/coco.names.txt"
labels = open(labels_path).read().strip().split("\n")

net = cv.dnn.readNetFromDarknet(cfg_path1,weights_path1)

names = net.getLayerNames()
Layers_Names = [names[i - 1] for i in net.getUnconnectedOutLayers()]

def load_img(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img
  
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
  
  
def plot_Bounding_Boxes(img, boxes, confidences, classIDs, idxs):
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x , y) = [boxes[i][0],boxes[i][1]]
            (w , h) = [boxes[i][2],boxes[i][3]]
    
            cv.rectangle(img,(x,y), (x+w,y+h), (0,0,255),2)
            cv.putText(img,"{}: {}".format(labels[classIDs[i]] ,confidences[i]) , (x,y-5) , cv.LINE_AA , 0.5 ,(255,255,255),2 )
    
    return img
  
def car_pipeline(img):
    #img = load_img(input_path)
    boxes, confidences, classIDs, idxs = run_inference(img,Layers_Names)
    out_img = plot_Bounding_Boxes(img, boxes, confidences, classIDs, idxs)
    #cv.imshow("Image",cv.cvtColor(img, cv.COLOR_BGR2RGB))
    #cv.waitKey(0)
    return out_img
  
def Create_Image2(input_path,output_path):
    img = load_img(input_path)
    new_img = car_pipeline(img)
    new_img = Image.fromarray(new_img.astype(np.uint8), 'RGB')
    new_img.save(output_path)
    
def Create_Video2(input_path,output_path):
    video_input = VideoFileClip(input_path)
    processed_video = video_input.fl_image(car_pipeline)
    %time  processed_video.write_videofile(output_path, audio=False)
