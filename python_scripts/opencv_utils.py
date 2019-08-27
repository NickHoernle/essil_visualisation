# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:45:49 2017

@author: Aditi
"""
from collections import deque
import numpy as np
import cv2
import math
import os
from scipy.spatial import distance as dist
import copy

COLORS = {  'plains_color': np.array([42,161,152], dtype=np.uint8),
            'wetlands_color': np.array([108,113,196], dtype=np.uint8),
            'desert_color': np.array([181,137,0], dtype=np.uint8),
            'jungle_color': np.array([211,54,130], dtype=np.uint8),
            'reservoir_color': np.array([232, 236, 237], dtype=np.uint8),
            'mountain_valley_color': np.array([232, 236, 237], dtype=np.uint8),
            'water_color': np.array([0,255,255], dtype=np.uint8),
            'logs_color': np.array([255,0,0], dtype=np.uint8) }


#to capture the position / cordinates of the mini video
def know_coordinates(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print (x ,",", y)

# finds the angle of the contour of water path and writes it into a file
def detect_water_encoding(min_video_frame,frame_num,directory):
    angle_matrix=[]
    #to decode the color information theo embeded
    hsv = cv2.cvtColor(min_video_frame, cv2.COLOR_BGR2HSV)
    # define range of green?? color in HSV
    lower_green = np.array([29,86,6])
    upper_green = np.array([64,255,255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(min_video_frame,min_video_frame, mask= mask)
   # cv2.imshow('res',res)
    #cv2.imwrite('frame.png',res)
    name = str(directory)+"\\water_path_angle"+str(frame_num)+".csv"
    rows, cols, channel = res.shape
    handle = open(name, 'wb')
    blank = np.zeros(res.shape,res.dtype)
    #get green and red channel and convert into useful information
    for row in range(0,rows):
        for col in range(0,cols):
            k = res[row,col]    #B=0 G=1 R=2
            #green channel
            green =  res.item(row,col,1)-120
            #red channel
            red = res.item(row,col,2)-120
            handle.write("\n")
            handle.write(str(row))
            handle.write(",")
            handle.write(str(col))
            handle.write(",")
            #Blue channel which was made 0 above
            handle.write(str(res.item(row,col,0)))
            handle.write(",")
            handle.write(str(green))
            handle.write(",")
            handle.write(str(red))
            handle.write(",")
            #all channels
            handle.write(str(k))
            """ red+120 <0 and green+120 < 0 first quadrant
                if green negative and red==0 y axis decide
                direction based on green negative or poistive
                if red < 0 ( third quadrant) add 180
                else check green negative
            """
            if(red + 120 < 20 and green + 120 < 20):
                angle = -1000
            elif(red==0):
                if((-1)*green < 0):#downwards y axis negative
                    angle =270
                    handle.write(",")
                    handle.write(str(angle))
                elif((-1)*green > 0): #upwards y axis positive
                    angle =90
                    handle.write(",")
                    handle.write(str(angle))
                else:
                    angle= -1000
            else:
                if(red < 0): #second quadrant left negative x axis
                    angle= 180 + math.degrees(math.atan2((-1)*green,red))
                    handle.write(",")
                    handle.write(str(angle))
                else: #positive x axis right
                    if((-1)*green < 0): #first quadrant upwards negative y axis
                        angle= 360 + math.degrees(math.atan2((-1)*green,red))
                        handle.write(",")
                        handle.write(str(angle))
                    else:#    fourth quadrant
                        angle=  math.degrees(math.atan2((-1)*green,red))
                        handle.write(",")
                        handle.write(str(angle))
            if(angle >= 0 and angle < 90):
                cv2.circle(blank,(col,row),1,(0,255,0),-1)
            elif(angle >=90 and angle <180):
                cv2.circle(blank,(col,row),1,(0,0,255),-1)
            elif(angle>= 180 and angle<270):
                cv2.circle(blank,(col,row),1,(255,0,0),-1)
            elif(angle>=270 and angle<=360):
                cv2.circle(blank,(col,row),1,(0,255,255),-1)
            angle_matrix.append([row,col,angle])
            #cv2.imshow("blank",blank)
    handle.close()
    return angle_matrix



#just identify water flow path for drawing graphs
def detect_water(min_video_frame):
    hsv = cv2.cvtColor(min_video_frame, cv2.COLOR_BGR2HSV)
    # define range of green/yellow color in HSV
    lower_green = np.array([29,86,6])
    upper_green = np.array([64,255,255])
    th3 = extract_boundary(min_video_frame,hsv,lower_green, upper_green,0)
    store = th3
    # morphing to get the skeletal structure/ medial line of the water flow
    size = np.size(th3)
    skel = np.zeros(th3.shape,np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while(not done):
        eroded = cv2.erode(th3,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(th3,temp)
        skel = cv2.bitwise_or(skel,temp)
        th3 = eroded.copy()

        zeros = size - cv2.countNonZero(th3)
        if zeros==size:
            done = True
    return store,skel

"""checks if a particular log is active or inactive, i.e. if
log touches water active else inactive"""
def check_logs(log,water):
    res = np.logical_and(log, water)
    ans = np.where(res == True)
    if len(ans[0])==0 and len(ans[1])==0:
        return "inactive"
    return "active"

def detect_logs(min_video_frame, filename, thresh_water):
    hsv = cv2.cvtColor(min_video_frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    th3 = extract_boundary(min_video_frame,hsv,lower_blue, upper_blue,0)

    #smooth the logs (current version very fat lines)
    image ,contours, heirarchy = cv2.findContours(th3,1,2)#cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Draw log contour + bonding rects
    colored = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    count =0
    black = np.zeros(colored.shape)
    centers=[]
    for contour in contours:
       # returns top left (x,y), width, height, and angle of rotation
       rect = cv2.minAreaRect(contour)
       box = cv2.boxPoints(rect)
       box = np.int0(box)
       """ rotated rectangle angle in opencv is calculated fro -0 to
       -90 in degrees as the angle the lowest vertex of the bounding
       rectangle makes with the horizontal counter clockwise direction, (the vertices are
       considered clockwise from the lowest vertex), width is the
       distance between second and third vertex, and height is the
       distance between first and second vertex.
       The following "if" converts the angle to o to -180 counterclockwise."""
       if(rect[1][0] < rect[1][1]):
           angle = rect[2] - 90
       else:
           angle = rect[2]
       filename.write(str(count))
       filename.write(",")
       filename.write(str(abs(angle)))
       filename.write(",")
       filename.write(str(rect[0][0]))
       filename.write(",")
       filename.write(str(rect[0][1]))
       filename.write(",")

       #to check if log is active/ inactive
       blank = np.zeros(thresh_water.shape,thresh_water.dtype)
       log = cv2.drawContours(blank,[contour],-1,(255,0,255),-1)
       res = check_logs(log,thresh_water)
       filename.write(res)
       filename.write(",")
       count= count+1
       image=cv2.drawContours(colored,[box],-1,(0,0,255),-1)
       #returns centroid of the bounding rectangle same  as contour centroid

       cx= int(rect[0][0])
       cy= int(rect[0][1])
       image = cv2.circle(black,(cx,cy),1,(0,255,0),-1)
       if(res is "active"):
           centers.append([cx,cy])
    return image,filename,centers


#def check_validity(frame, frame_number, handle_valid):
#    handle_valid.write("\n")
#    rows,cols, channels = frame.shape
#    print frame.shape
#    for row in range(0,rows):
#        for col in range(0,cols):
#            if(row>= 5 and row <=9 and col >=215 and col<=219):
#                k = frame[row,col]
#                handle_valid.write(str(row))
#                handle_valid.write(",")
#                handle_valid.write(str(col))
#                handle_valid.write(",")
#                handle_valid.write(str(frame_number))
#                handle_valid.write(",")
#                handle_valid.write(str(k))
#                handle_valid.write(",")
#    return handle_valid


def extract_boundary(original,hsv_image, lower, upper, flag):
    # need end points of the boundary too
    mask = cv2.inRange(hsv_image, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(original,original,mask=mask)
    #boundaries in gray scale
    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding and gaussian filtering  to make the logs white and the background black for better detection
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #logs will be white in th3
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if(flag==1):
        black, extLeft, extRight, cx,cy = find_contour(th3,original)
        return black,extLeft,extRight,cx,cy
    return th3


def find_contour(th3,original):
    black = np.zeros(original.shape,original.dtype)
    image ,contours, heirarchy = cv2.findContours(th3,1,2)
    for contour in contours:
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        black = cv2.circle(black,(cx,cy),3,(255,255,255),-1)
        extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
        extRight = tuple(contour[contour[:, :, 0].argmax()][0])
        black = cv2.circle(black,extLeft,3,(255,255,255),-1)
        black = cv2.circle(black,extRight,3,(255,255,255),-1)
    return black,extLeft,extRight,cx,cy

def fix_color(matrix, lower, upper, new_color, roll=2):
    mask = np.zeros_like(matrix[:,:,0])
    mask += matrix[:,:,0]<=upper[0]
    mask += matrix[:,:,0]>=lower[0]
    mask += matrix[:,:,1]<=upper[1]
    mask += matrix[:,:,1]>=lower[1]
    mask += matrix[:,:,2]<=upper[2]
    mask += matrix[:,:,2]>=lower[2]

    matrix[mask==6] = new_color
    matrix[np.roll(mask, roll, axis=0)==6] = new_color
    matrix[np.roll(mask, -roll, axis=0)==6] = new_color
    matrix[np.roll(mask, roll, axis=1)==6] = new_color
    matrix[np.roll(mask, -roll, axis=1)==6] = new_color
    return matrix

def mark_boundaries():
     boundary_ends=[]
     center_biomes =[]

     convert_boundaries = cv2.imread("CW-BiomeBoundary.png")
     hsv = cv2.cvtColor(convert_boundaries, cv2.COLOR_BGR2HSV)

     # mountain valley
     lower = np.array([0,100,0])
     upper = np.array([5,255,5])
     convert_boundaries = fix_color(convert_boundaries, lower, upper, COLORS['mountain_valley_color'])

     #Plains
     lower = np.array([170,0,0])
     upper = np.array([255,5,5])
     convert_boundaries = fix_color(convert_boundaries, lower, upper, COLORS['plains_color'])

     #jungle - megenta
     lower=np.array([100, 0, 100])
     upper=np.array([255,5,255])
     convert_boundaries = fix_color(convert_boundaries, lower, upper, COLORS['jungle_color'])

     #wetlands
     lower=np.array([0, 150, 150])
     upper=np.array([5,255,255])
     convert_boundaries = fix_color(convert_boundaries, lower, upper, COLORS['wetlands_color'])

     #reservoir
     lower=np.array([135, 135, 0])
     upper=np.array([255,255,5])
     convert_boundaries = fix_color(convert_boundaries, lower, upper, COLORS['reservoir_color'])

     #desert
     lower = np.array([0,0,170])
     upper = np.array([5,5,255])
     convert_boundaries = fix_color(convert_boundaries, lower, upper, COLORS['desert_color'])

     temp = np.float64(convert_boundaries)
     return temp, boundary_ends, center_biomes

def find_biome_sector(shape,dtype,start,end,x,y):
    blank = np.zeros(shape,dtype)
    pts=np.array([start,end,[x,y]])
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    upper = dist.euclidean(box[0],box[1])
    lower = dist.euclidean(box[1], box[2])
    #if longest side is not necessarily the 0 and 1 coordinates
    if lower > upper:
        temp = lower
        lower= upper
        upper= temp
    box = np.int0(box)
    cv2.drawContours(blank,[box],0,(255,255,255),2)
    filled = cv2.fillPoly(blank, [pts], color=(255,255,255))
    return filled, lower, upper

#def find_long_side(x,y,start,end):
#    dist1 = dist.euclidean([x,y],start)
#    dist2 = dist.euclidean([x,y],end)
#    limit = abs(dist1-dist2)
#    if(dist1 >= dist2):
#        return dist1,limit
#    else:
#        return dist2,limit

def log2log(biome,log_centers,flag_logs):
    i =0
    for x,y in log_centers:
        if(biome[y,x] == 255 and flag_logs[i]==0):
            join =[x,y]
            break
        i=i+1
    if(i >= len(log_centers)):
         join = [-1000,-1000]
    return join

#def customsort(log_centers,logs):
#    log_x=[]
#    log_y=[]
#    for log in logs:
#        x,y = log_centers[log]
#        log_x.append(x)
#        log_y.append(y)
#    sorted_logs = np.lexsort((log_y,log_x))
#    return_logs = [(log_x[i],log_y[i]) for i in sorted_logs]
#    print return_logs
#    return return_logs

def draw_graph(boundaries,skeleton,log_centers,boundary_ends,center_biome):
    log_list = []
    flag=0
    for log in log_centers:
        #a flag along with the coordinates to indcate log=0 or wall=1
        log_list.append([])
        x,y = log
        log_list[flag].append([(x,y),0])
        flag=flag+1

    flag_logs =np.zeros(len(log_centers))
    blank = np.zeros(boundaries.shape, boundaries.dtype)
    flag =0
    for x,y in log_centers:
            flag_logs[flag]=1
            cv2.circle(blank,(x,y),3,[0,0,255],-1)
            count=0
            for i in range(0,len(boundary_ends),2):
                filled_biome, lower, upper = find_biome_sector(skeleton.shape,skeleton.dtype,boundary_ends[i],boundary_ends[i+1],x,y)
                res = cv2.bitwise_and(np.float32(filled_biome),np.float32(skeleton))
                #cv2.imshow("result",res)
                if(cv2.countNonZero(res) > lower):#lower and cv2.countNonZero(res) <= upper):  #threshhold of  pixels
                    #indices of logs in the same sector=1 else 0
                    join = log2log(filled_biome,log_centers,flag_logs)
                    x_join , y_join = join
                    #separate join and if it is not -1000 then draw line from log to end
                    if(x_join!=-1000 and y_join!=-1000):
                        log_list[flag].append([(x_join,y_join),0])
                        cv2.line(blank.astype(np.int32),(x,y),(x_join,y_join),[0,255,255],1)
                       # cv2.imshow("middle",blank)#check for upper asending may not work
                    else: #(other check can be checking if point of intersection is between which boundaries ends and join for adjacent boundary ends)
                        #check point of intersection in which sector this or neighbouring
                        if(count==1):
                            center_count = check_bound_intersection(boundaries,res,boundary_ends,1)
                        elif(count==5):
                            center_count = check_bound_intersection(boundaries,res,boundary_ends,2)
                        else:
                            center_count = count
                        center_count = int(center_count)
                        if(center_count!=13):     # if 13 no line to be drawn
                            cb = (int(center_biomes[center_count][0]), int(center_biomes[center_count][1]))
                            log_list[flag].append([cb,1])
                            cv2.line(blank,(x,y),cb,[0,255,255],1)
                count=count+1
            flag=flag+1
    #checking cycles without directions is difficult
    #so for each two logs that are neighbours of each other measure distance from the common wall
    #check for cycles remove redundant edges  (can i remove longest edge??)

    #log_list = remove_redundant_edges(log_list)

    #convert into alphabets and write into a file framewise
    #blanknew = cv2.add(boundaries,blank)
   # cv2.imshow('blanknew',blanknew)
    return log_list


def remove_redundant_edges(log_list):
    count =0
    for log in log_list:
        neighbour_logs = find_neighbour_log(log)
        head = neighbour_logs.popleft()
        while (len(neighbour_logs)!=0):
            tail = neighbour_logs.popleft()
            index = locate_tail(tail,log_list)

            common = findcommon(log_list[count],log_list[index])
            for each_common in common:
                discard = find_discard(head,tail,each_common)
                if(discard =="head"):
                    log_list[count].remove(each_common)
                else:
                    log_list[index].remove(each_common)
        count =count+1
    return log_list


def locate_tail(tail, log_list):
    all_heads = [item[0] for item in log_list]
    for i in range(0,len(all_heads)):
        if(all_heads[i]==tail):
            return i

def find_discard(log1,log2,common):
    [log1_x,log1_y], log1_flag = log1
    [log2_x,log2_y], log2_flag = log2
    [common_x,common_y], common1_flag = common
    log1_common = dist.euclidean([log1_x,log1_y],[common_x,common_y])
    log2_common = dist.euclidean([log2_x,log2_y],[common_x,common_y])
    if(log1_common > log2_common):
        return "head"
    else:
        return "tail"


def findcommon(log1, log2):
    temp1 = copy.deepcopy(log1)
    temp2 = copy.deepcopy(log2)
    del temp1[0]
    del temp2[0]
    return [x for x in temp1 if x in temp2]

def find_neighbour_log(log):
    d = deque()
    for element in log:
        (x,y),flag = element
        if (flag==0):
           d.append([(x,y),flag])
    return d

def check_bound_intersection(boundaries,res,boundary_ends,flag):
    #may not work when  some times the water doesnt intersect the walls like in desert goes little out side
     boundaries_thresh = cv2.cvtColor(np.float32(boundaries),cv2.COLOR_BGR2GRAY)
     res = cv2.bitwise_and(boundaries_thresh,np.float32(res))
     index = np.where(res)
     index_x = index[1]
     index_y = index[0]
     if(flag==1):
         start = 0
         finish = 5
     elif(flag==2):
         start=8
         finish=13
     for ends in range(start, finish,2):
         if(ends>=0 and ends<=5):
             lower= boundary_ends[ends]
             upper = boundary_ends[ends+1]
         elif(ends >=8  and ends <=13):
             lower= boundary_ends[ends +1]
             upper = boundary_ends[ends]
        # if(x >= lower[0] and y>=lower[1] and x<= upper[0] and y<=upper[1]):
         check = np.where(np.greater_equal(index_x,lower[0]) & np.less_equal(index_x,upper[0]) & np.greater_equal(index_y,lower[1]) & np.less_equal( index_y,upper[1]))
         if(len(check[0])!=0):
             return ends/2   # only first point may not be a good idea

             #may be calculate maximum coverage

     return 13   # some dummy value if intersection doesnt fall in the range


#def check_angle(angle_matrix, row,col):
#    for row_elem, col_elem,angle in angle_matrix:
#         if row == row_elem and col == col_elem:
#            return angle
#    return 0

def add_waterfall(boundaries, boundary_ends, center_biomes):
    water_fall_start = boundary_ends[6]
    water_fall_end = boundary_ends[7]
    x_start= water_fall_start[0]
    y_start = water_fall_start[1]
    x_end = water_fall_end[0]
    y_end = water_fall_end[1]
    x_mid = (x_start + x_end) / 2
    y_mid = (y_start + y_end) / 2
    center_biomes.insert(3,((x_mid,y_mid)))
    boundaries = cv2.line(boundaries,boundary_ends[6],boundary_ends[7],[255,255,255],3)
    return boundaries,center_biomes


def biome_alpha(index):
    if(index==0):
        return "D"
    elif(index==1):
        return "MV"
    elif(index==2):
        return "P"
    elif(index==3):
        return "WF"
    elif(index==4):
        return "J"
    elif(index==5):
        return "R"
    elif(index==6):
        return "W"


def write_log_list(image,file_write,log_list,frame, center_biomes):
    file_write.write("\n"+str(frame))
    file_write.write(",")
    unique_logs=[item[0] for item in log_list]
    for lists in log_list:
        head = lists[0]
        head_pts, flag = head
        if(flag==0):
            file_write.write(str(head_pts)+": log"+str(unique_logs.index(head))+"->")
        for tail in range(1,len(lists)):
            tail_pts, flag =  lists[tail]
            if(flag==0):
                file_write.write(str(tail_pts)+": log"+str(unique_logs.index(lists[tail]))+"->")
            else:
                index = center_biomes.index(tail_pts)
                alpha= biome_alpha(index)
                file_write.write(str(tail_pts)+":"+ alpha+ "->" )
            cv2.line(image,head_pts,tail_pts,[75,0,130],3)
    cv2.imshow("finalize",image)
    return file_write,image


# filename = 'C:\\Aditi\\phd\\Log_Files\\2017-05-05\\theo emails\\17-43-36-May_17_Highschool-miniVid-PlantCreatureInfo\\17-43-36-May_17_Highschool-miniVid-PlantCreatureInfo\\0-Section_One-17-43-36-502.mov'
# #filename = 'C:\\Aditi\\phd\\Log_Files\\2017-05-05\\theo emails\\june 6 email\\16-21-27-May_17_Highschool-StoicBabyMigrate\\16-21-27-May_17_Highschool-StoicBabyMigrate\\0-Section_One-16-21-27-528.mov'
# cap = cv2.VideoCapture(filename)
# ex_filename = filename.strip(".mov")
# directory=ex_filename
# ex_filename = str(ex_filename)+".csv"
#
# handle = open(ex_filename, 'w')
# handle_valid= open('validity_file.csv','wb')
# """ per frame angle should be stored, so for each video file
# for each frame create a water_path angles file giving water contour angles
# """
# name = str(directory)+"\\adjacency_list.txt"
# file_writer= open(name,'w')
# handle.write("frame position")
#
# for i in range(0,8):
#     handle.write(",")
#     handle.write("index of log")
#     handle.write(",")
#     handle.write("angle with horizontal")
#     handle.write(",")
#     handle.write("centroid x")
#     handle.write(",")
#     handle.write("centroid y")
#     handle.write(",")
#     handle.write("status: active/inactive")
#
# cv2.namedWindow('video',0)
# boundaries, boundary_ends, center_biomes = mark_boundaries()
# complete_boundary,center_biomes = add_waterfall(boundaries,boundary_ends, center_biomes)
# while(cap.isOpened()):
#     flag, frame = cap.read()
#     if flag:
#         #display only mini video
#         min_video = frame[864:1080,0:384]
#
#         cv2.imshow('video',min_video)
#         handle.write("\n")
#         temp = frame[864:1080,0:384]
#
#         #break
#         # 1= will be frame number of next frame
#         pos_frame = cap.get(1)
#         #handle_valid=check_validity(min_video,pos_frame-1,handle_valid)
#
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         angle_matrix = detect_water_encoding(temp,(pos_frame-1),directory)
#         handle.write(str(pos_frame-1))
#         handle.write(",")
#         thresh_water, skeleton = detect_water(min_video)
#         logs,handle,centers = detect_logs(min_video, handle,thresh_water)
#         #cv2.imwrite("logs.png",logs)
#         color_water = cv2.cvtColor(skeleton,cv2.COLOR_GRAY2BGR)
#         water = np.float64(color_water)
#         #cv2.imwrite('snap.png',thresh_water)
#         dst = cv2.add(water,logs)
#         dst = cv2.add(complete_boundary, dst)
#
#         log_list = draw_graph(dst,skeleton,centers,boundary_ends,center_biomes)
#         file_writer,image = write_log_list(dst,file_writer,log_list,str(pos_frame-1),center_biomes)
#         #The frame is ready and already captured
#         cv2.imshow('video', image)
#
#
# #        break #added to process just one frame
#         cap.set(3,240)
#         cap.set(4,640)
#         cv2.setMouseCallback('video',know_coordinates)
#         cv2.waitKey(1000)
#     else:
#         # The next frame is not ready, so we try to read it again
#         cap.set(1, pos_frame-1)
#         print ("frame is not ready")
#         # It is better to wait for a while for the next frame to be ready
#         cv2.waitKey(1000)
#
#     if cv2.waitKey(10) == 27:
#         break
#     if cap.get(1) == cap.get(7):
#         # If the number of captured frames is equal to the total number of frames,
#         # we stop
#         break
# #handle_valid.close()
# file_writer.close()
# handle.close()
# cap.release()
# cv2.destroyAllWindows()
