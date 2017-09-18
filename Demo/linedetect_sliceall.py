import numpy as np
import cv2
import operator
from matplotlib import pyplot as plt
import os, errno
import glob, os

mainfolder = 'detectresult/image14/'

if not os.path.exists('detectresult/image14/VerticalSlice'):
    os.makedirs('detectresult/image14/VerticalSlice')

def DeleteColor(imgPath):    
    img= cv2.imread(imgPath)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask!=0)] = 255

    cv2.imwrite(imgPath, output_img)

def SharpenImage(img): 

    #Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros( (100,100), np.float32)
    kernel[2,2] = 2.0   #Identity, times two! 

    #Create a box filter:
    boxFilter = np.ones( (100,100), np.float32) / 10000.0

    #Subtract the two:
    kernel = kernel - boxFilter

    #Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.

    custom = cv2.filter2D(img, -1, kernel)

    return custom

# Denoising
for item in glob.glob("image/*.jpg"):
    print("remove printer")
    DeleteColor(item)

    print("denoise " + item)
    
    img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)   
    img = SharpenImage(img)
  
    dst = cv2.fastNlMeansDenoising(img,None,10,7,21)
   
    cv2.imwrite('%s\\%s' % ("denoise", item.split('\\')[1]), dst)

#Detect Line 
for item in glob.glob("denoise/*.jpg"):
    #print("Image: %s" % (item))
    img = cv2.imread(item)
    copy_image = cv2.imread(item)

    #Get image shape
    height, width, channels = img.shape

    #print(height, width, channels) 
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 10
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, minLineLength=120, maxLineGap=5)   
    
    twod_list = []
    rangeValue = lines.size/4
    
    for index in range(0, int(rangeValue)):
        new  = []
        new.append(lines[index][0][0])
        new.append(lines[index][0][1])
        new.append(lines[index][0][2])
        new.append(lines[index][0][3])
        twod_list.append(new)

    twod_list = sorted(twod_list,key=lambda x: x[1])
    #print(twod_list)
    
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200)
    #print(lines.size)
    sliceArray = []
    verticalArray = []

    for line in twod_list:       
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        #draw the horizatal line
        if abs(y1 - y2) < 10 and abs(x1 - x2) > 10:                
            #cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

            #Add the valid Y axis point to array
            if len(sliceArray) == 0: 
                sliceArray.append(y1)
            elif abs(sliceArray[len(sliceArray)-1] - y1) > 40:
                sliceArray.append(y1)     

    for yAxis in sliceArray:
        cv2.line(copy_image,(0,yAxis),(width,yAxis),(0,0,255),2)


    #draw the vertical line
    twod_list = sorted(twod_list,key=lambda x: x[0])  
    for line in twod_list:       
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        
        if abs(y1 - y2) > 50 and abs(x1 - x2) < 10:          
            if len(verticalArray) == 0: 
                verticalArray.append(x1)
            elif abs(verticalArray[len(verticalArray)-1] - x1) > 40:
                verticalArray.append(x1)

    for xAxis in verticalArray:
        cv2.line(copy_image,(xAxis,0),(xAxis,height),(0,0,255),2)

    cv2.imwrite('%s\\%s' % ("detectresult", item.split('\\')[1]), copy_image)
   
    for index in range(len(sliceArray)-1): 
        for verticalIndex in range(len(verticalArray)-1): 
            height = sliceArray[index+1]
            width = verticalArray[verticalIndex+1]

            crop_img = copy_image[sliceArray[index]:height, verticalArray[verticalIndex]:width]

            filename = str(index) + '_' + str(verticalIndex) + '.jpg'
            cv2.imwrite(mainfolder + filename, crop_img)