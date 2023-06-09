import cv2
import argparse
import orien_lines
import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
import pandas as pd
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy
import numpy as np

list1= []
list2 = []

ap = argparse.ArgumentParser()

ap.add_argument('-d','--display',dest = 'display',type = int, default = 1,
                help = 'Display the detected images using OpenCV')

args = vars(ap.parse_args())

detection_graph,sess = detector_utils.load_inference_graph()

def save_data(no_of_time_hand_detected, no_of_time_hand_crossed):

    try:   
        today = date.today()
        today=str(today)
        #loc = (r'C:\Users\rahul.tripathi\Desktop\result.xls') 
      
        rb = xlrd.open_workbook('result.xls')
        sheet = rb.sheet_by_index(0) 
        sheet.cell_value(0, 0) 
      
         
        #print(sheet.nrows)
        q=sheet.cell_value(sheet.nrows-1,1)
        
        rb = xlrd.open_workbook('result.xls') 
        #rb = xlrd.open_workbook(loc) 
        wb=copy(rb)
        w_sheet=wb.get_sheet(0)
        
        if q==today:
            w=sheet.cell_value(sheet.nrows-1,2)
            e=sheet.cell_value(sheet.nrows-1,3)
            w_sheet.write(sheet.nrows-1,2,w+no_of_time_hand_detected)
            w_sheet.write(sheet.nrows-1,3,e+no_of_time_hand_crossed)
            wb.save('result.xls')      
        else:
            w_sheet.write(sheet.nrows,0,sheet.nrows)
            w_sheet.write(sheet.nrows,1,today)
            w_sheet.write(sheet.nrows,2,no_of_time_hand_detected)
            w_sheet.write(sheet.nrows,3,no_of_time_hand_crossed)
            wb.save('result.xls')
    except FileNotFoundError:
        today = date.today()
        today=str(today)
         

        # Workbook is created 
        wb = Workbook() 

        # add_sheet is used to create sheet. 
        sheet = wb.add_sheet('Sheet 1') 

        sheet.write(0, 0, 'Sl.No')
        sheet.write(0, 1, 'Date') 
        sheet.write(0, 2, 'Number of times hand detected') 
        sheet.write(0, 3, 'Number of times hand crossed') 
        m=1
        sheet.write(1, 0, m)
        sheet.write(1, 1, today) 
        sheet.write(1, 2, no_of_time_hand_detected) 
        sheet.write(1, 3, no_of_time_hand_crossed) 

        wb.save('result.xls')

if __name__ == '__main__':

    score_thresh = 0.8

    vs = VideoStream(0).start()

    Orientation = 'bt'

    #input("Enter the orientation of hand progression ~ lr,rl,bt,tb :")

    Line_Perc1 = float(15)

    Line_perc2 = float(30)

    num_hands_detect = 2

    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None,None)

    cv2.namedWindow("Detection",cv2.WINDOW_NORMAL)

    def count_no_times(l:list):
        x = y = cnt=0
        for i in l:
            x = y
            y = i
            if x==0 and y==1:
                cnt = cnt+1
        return cnt
    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)

            if im_width == None:
                im_height,im_width = frame.shape[:2]

            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            except:
                print("This frame can't be converted")

            boxes,scores,classes = detector_utils.detect_objects(frame,detection_graph,sess)

            Line_Position2=orien_lines.drawsafelines(frame,Orientation,Line_Perc1,Line_perc2)

            a,b = detector_utils.draw_box_on_image(num_hands_detect,score_thresh,scores,boxes,classes,im_width,im_height,frame,Line_Position2,Orientation)

            list1.append(a)
            list2.append(b)

            no_of_time_hand_detected= no_of_time_hand_crossed =0

            ##calculate frame per second(FPS)

            num_frames += 1
            elapsed_time = (datetime.datetime.now()-start_time).total_seconds()

            fps = num_frames /elapsed_time

            if args["display"]:

                ####Display fps on frame

                detector_utils.draw_text_on_image('FPS : '+ str("{0:.2f}".format(fps)),frame)
                cv2.imshow('Detection',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break
        no_of_time_hand_detected = count_no_times(list2)
        no_of_time_hand_crossed = count_no_times(list1)

        save_data(no_of_time_hand_detected,no_of_time_hand_crossed)
        print("Average FPS:" + str("{0:.2f}".format(fps)))
        print(list2)

    except KeyboardInterrupt:
         no_of_time_hand_detected = count_no_times(list2)
         no_of_time_hand_crossed = count_no_times(list1)
         today = date.today()

         save_data(no_of_time_hand_detected,no_of_time_hand_crossed)
         #print("Average FPS:" + str("{0:.2f}".format(fps)))
         

 



             
                

                                                  







          







