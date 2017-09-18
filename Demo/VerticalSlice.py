import numpy as np
import cv2
import operator
from matplotlib import pyplot as plt
import os, errno
import glob, os

mainfolder = 'detectresult/image14'

if not os.path.exists('detectresult/image14/VerticalSlice'):
    os.makedirs('detectresult/image14/VerticalSlice')

if not os.path.exists('detectresult/image14/VerticalSlice/Slices'):
    os.makedirs('detectresult/image14/VerticalSlice/Slices')        

if not os.path.exists('detectresult/image14/VerticalSlice/Edge'):
    os.makedirs('detectresult/image14/VerticalSlice/Edge')        

for item in glob.glob(mainfolder + '/*.jpg'):
    gray = cv2.imread(item)
    img = cv2.imread(item)

    edges = cv2.Canny(gray,20,40,apertureSize = 3)
    cv2.imwrite('detectresult/image14/VerticalSlice/Edge/' + item.split('\\')[1], edges)    

    height, width, channels = gray.shape  

    print(item)
    
    minLineLength=10
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=10,lines=np.array([]), minLineLength=0,maxLineGap=0)

    twod_list = [] 

    a,b,c = lines.shape
    for i in range(a):   

        if abs(lines[i][0][1] - lines[i][0][3]) > 10:
            new  = []
            new.append(lines[i][0][0])
            new.append(lines[i][0][1])
            new.append(lines[i][0][2])
            new.append(lines[i][0][3])

            twod_list.append(new)              

    twod_list = sorted(twod_list,key=lambda x: x[0])

    sliceArray = []
    for line in twod_list:       
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        if abs(y1 - y2) > 5 and abs(x1 - x2) < 10:
            cv2.line(img, (x1, 0), (x1, height), (0, 0, 255), 2)   
            cv2.imwrite('detectresult/image14/VerticalSlice/' + item.split('\\')[1], img)    

            if len(sliceArray) == 0: 
                sliceArray.append(x1)
            elif abs(sliceArray[len(sliceArray)-1] - x1) > 40:
                sliceArray.append(x1)
    
    #crop images
    # NOTE: its img[y: y + h, x: x + w]
    if len(sliceArray) > 3:
        crop_img = gray[0:height, 0:sliceArray[0]]
        
        path = str(mainfolder + '/VerticalSlice/Slices/')      
        filename = item.split('\\')[1].split('.')[0] + '_' + "1.jpg"
       
        cv2.imwrite(path + filename, crop_img)

        for index in range(len(sliceArray)-1):              
            pointX = sliceArray[index]
            height = height
            pointY = 0
            width =  sliceArray[index] + (sliceArray[index + 1] - sliceArray[index]) 
            
            crop_img = gray[pointY:height, pointX:width] # Crop from x, y, w, h -> 100, 200, 300, 400

            filename = item.split('\\')[1].split('.')[0] + '_' + ("%d.jpg" % (index+2))
         
            cv2.imwrite(path + filename, crop_img)            

        #save the last slice of the image
        pointX = sliceArray[len(sliceArray)-1]
        height = height
        pointY = 0
        width = sliceArray[len(sliceArray)-1] + (height - len(sliceArray)-1)
        
        crop_img = gray[pointY:height, pointX:width] # Crop from x, y, w, h -> 100, 200, 300, 400
        filename = item.split('\\')[1].split('.')[0] + '_' + ("%d.jpg" % (len(sliceArray)+1))
        cv2.imwrite(path + filename, crop_img)

