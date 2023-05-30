##importing the libraries

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from utils import alertcheck
detection_graph = tf.Graph()

TRAINED_MODEL_DIR = "frozen_graphs"

PATH_TO_CKPT = TRAINED_MODEL_DIR + "/ssd5_optimized_inference_graph.pb"

PATH_TO_LABELS = TRAINED_MODEL_DIR + "/hand_label_map.pbtxt"

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes = NUM_CLASSES , use_display_name = True)

category_index = label_map_util.create_category_index(categories)

a=b=0

##load the frozer inference graph into the memory

def load_inference_graph():

    ##load frozen tensorflow model into memory

    print(">===== loading frozen graph into memory ======")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')
        sess = tf.Session(graph= detection_graph)
    print("===== inference graph loaded.")
    return detection_graph,sess

def draw_box_on_image(num_hands_detect,score_thresh,scores,boxes,classes,im_width,
                      im_height,image_np,Line_Position2,Orientation):
    
    focallength = 300   ###Determined using a piece of paper as a marker and code is mentioned in distance to camera

    avg_width = 4.0   ### avg hand width 

    global a,b
    hand_cnt= 0
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    for i in range(num_hands_detect):

        if (scores[i] > score_thresh):

            if classes[i] == 1:
                id = "hand"

            if classes[i] == 2:
                id = "gloved_hand"
                avg_width = 3.0 ## To compensate for the bbox size change

            if i == 0:
                color = color0
            else:
                color = color1                  ###Normalizing the bbox coordinates(xmin,ymin,xmax,ymax)
            (left,right,top,bottom) = (boxes[i][1]*im_width,boxes[i][3]*im_width,
                                       boxes[i][0]*im_height,boxes[i][2]*im_height)
            p1 = (int(left), int(top))
            p2= (int(right), int(bottom))

            dist = distance_to_camera(avg_width,focallength,int(right-left))

            if dist:
                hand_cnt = hand_cnt+1

            cv2.rectangle(image_np,p1,p2,color,3,1)

            cv2.putText(image_np,'hand '+ str(i)+':'+id,(int(left),int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            cv2.putText(image_np,"confidence"+ str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            cv2.putText(image_np,'distance_from_camera'+ str("{0:.2f}".format(dist)),
                        (int(im_width*0.65),int(im_height*0.9+30*i)),
                        cv2.FONT_HERSHEY_COMPLEX,0.5,color,2)

            a = alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)

        if hand_cnt == 0:
            b = 0
        else:
            b = 1
    return a,b   
"""
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np,Line_Position2,Orientation):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 575
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a,b
    hand_cnt=0
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    for i in range(num_hands_detect):
        
        if (scores[i] > score_thresh):
        
            #no_of_times_hands_detected+=1
            #b=b+1
            #b=1
            #print(b)
            if classes[i] == 1: 
                id = 'hand'
                #b=1
            
                
            if classes[i] == 2:
                id ='gloved_hand'
                avg_width = 3.0 # To compensate bbox size change
                #b=1
            
            if i == 0: color = color0
            else: color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            dist = distance_to_camera(avg_width, focalLength, int(right-left))
            
            if dist:
                hand_cnt=hand_cnt+1           
            cv2.rectangle(image_np, p1, p2, color , 3, 1)
            

            cv2.putText(image_np, 'hand '+str(i)+': '+id, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.putText(image_np, 'distance from camera: '+str("{0:.2f}".format(dist)+' inches'),
                        (int(im_width*0.65),int(im_height*0.9+30*i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
           
            a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)
        if hand_cnt==0 :
            b=0
            #print(" no hand")
        else:
            b=1
            #print(" hand")
            
    return a,b        
"""            

   
####To frames per second on the image
def draw_text_on_image(fps,image_np):
    cv2.putText(image_np,fps,(20,50),cv2.FONT_HERSHEY_COMPLEX,2,(142,255,9),2)

##compute & return the distance from the hand to the camera
def distance_to_camera(knownWidth,focalLength,pixelWidth):
    return (knownWidth*focalLength) / pixelWidth

def detect_objects(image_np,detection_graph,sess):

    ##Define input & output tensors for the detection graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    ##Each box represent the part of the image where the object is getting detected

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    ##Confidence represents the score with which objects getting detected
    ##This should be visible on the image if the object is getting detected

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np,axis= 0)

    (boxes,scores,classes,num) = sess.run([detection_boxes,detection_scores,detection_classes,num_detections],feed_dict={image_tensor:image_np_expanded})

    return np.squeeze(boxes),np.squeeze(scores),np.squeeze(classes)            
    







               













            











