#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 23:49:16 2022

@author: dushyant
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 18:19:57 2022

@author: dushyant
"""


# Program To Read video
# and Extract Frames
import cv2 as cv
import numpy as np
import dlib


# Function to extract frames
def FrameCapture(path):
    vidObj = cv.VideoCapture(path)
    width = int(vidObj.get(3))
    height = int(vidObj.get(4))
    frame_size = (width, height)
    fps = vidObj.get(5)
    output = cv.VideoWriter('/Data/Data2OutputTri.mp4',cv.VideoWriter_fourcc(*'MJPG'),20,frame_size)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    while success:
        success, img = vidObj.read()
        img3 = img.copy()
        img4 = img.copy()
        img5 = img.copy()
        img6 = img.copy()
        img7 = img.copy()
        imgray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        im2gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        mask = np.zeros_like(imgray)

        fake = np.zeros_like(img)
        fake2 = np.zeros_like(img)

        #Find face fiducials
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces = detector(imgray)


        size = img.shape
        rect = (0, 0, size[1], size[0])
        face = faces[0]

        landmarks = predictor(imgray, face)
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x,y))
            
        size = img.shape
        rect2 = (0, 0, size[1], size[0])
        faces2 = detector(im2gray)

        face2 = faces[1]
        #Face 2
        landmarks2 = predictor(imgray, face2)
        landmarks_points2 = []
        for n in range(68):
            x = landmarks2.part(n).x
            y = landmarks2.part(n).y
            landmarks_points2.append((x,y))

            
        landmarks_points = np.asarray(landmarks_points)
        landmarks_points2 = np.asarray(landmarks_points2)
        
        if (max(landmarks_points[:,0]) <  min(landmarks_points2[:,0])):
            points = np.asarray(landmarks_points, np.int32)
            points2 = np.asarray(landmarks_points2, np.int32)
            
        else:
            points = np.asarray(landmarks_points2, np.int32)
            points2 = np.asarray(landmarks_points, np.int32)


        subdiv  = cv.Subdiv2D(rect)
        points = np.asarray(points,np.float32)
        for p in points:
            subdiv.insert(p)
        b = subdiv.getTriangleList()
        linecolor = (250,0,0)

        subdiv  = cv.Subdiv2D(rect2)
        points2 = np.asarray(points2,np.float32)
        for p in points2:
            subdiv.insert(p)
        b2 = subdiv.getTriangleList()
        linecolor = (0,250,0)

        '''
        for i in range(b.shape[0]):
            cv.line(img, (int(b[i][0]),int(b[i][1])), (int(b[i][2]),int(b[i][3])), linecolor, 1)
            cv.line(img, (int(b[i][4]),int(b[i][5])), (int(b[i][2]),int(b[i][3])), linecolor, 1)
            cv.line(img, (int(b[i][0]),int(b[i][1])), (int(b[i][4]),int(b[i][5])), linecolor, 1)
        '''
        #cv.imshow('face1',img)
 
        '''
        for i in range(b2.shape[0]):
            cv.line(img, (int(b2[i][0]),int(b2[i][1])), (int(b2[i][2]),int(b2[i][3])), linecolor, 1)
            cv.line(img, (int(b2[i][4]),int(b2[i][5])), (int(b2[i][2]),int(b2[i][3])), linecolor, 1)
            cv.line(img, (int(b2[i][0]),int(b2[i][1])), (int(b2[i][4]),int(b2[i][5])), linecolor, 1)
        '''
        points = np.asarray(points,dtype= int)
        points2 = np.asarray(points2,dtype= int)

        #cv.imshow('face2', img)
        #check the triangle correspondance
        ##########################################
        ##finding index
        def find_ind(im_tr_pt, points):
            ind = np.where((points == im_tr_pt).all(axis=1))
            ind = ind[0][0]
            return ind


        #finding vertices of each triangle which will be sent to find indices of each vertex
        print('b_shape', b.shape[0])
        tr1_index = np.zeros([b.shape[0], 3])
        for i in range (b.shape[0]):
            im1_tr_pt1 = b[i][:2]
            im1_tr_pt1_index = find_ind(im1_tr_pt1,points)
            tr1_index[i][0] = im1_tr_pt1_index
            
            im1_tr_pt2 = b[i][2:4]
            im1_tr_pt2_index = find_ind(im1_tr_pt2,points)
            tr1_index[i][1] = im1_tr_pt2_index
            
            im1_tr_pt3 = b[i][4:6]
            im1_tr_pt3_index = find_ind(im1_tr_pt3,points)
            tr1_index[i][2] = im1_tr_pt3_index

        #print(tr1_index[101:112])
        tr1_index = np.asarray(tr1_index, dtype = int)


        #print('triangle vertex1', points[tr1_index[29]][0])


        #finding rectangle bounding triangle
        def bounding_rect(triangle):
            xmin = min(points[triangle][0][0], points[triangle][1][0], points[triangle][2][0])
            xmax = max(points[triangle][0][0], points[triangle][1][0], points[triangle][2][0])

            ymin = min(points[triangle][0][1], points[triangle][1][1], points[triangle][2][1])
            ymax = max(points[triangle][0][1], points[triangle][1][1], points[triangle][2][1])
            return(xmin, xmax, ymin, ymax)


        def bounding_rect2(triangle):
            xmin = min(points2[triangle][0][0], points2[triangle][1][0], points2[triangle][2][0])
            xmax = max(points2[triangle][0][0], points2[triangle][1][0], points2[triangle][2][0])

            ymin = min(points2[triangle][0][1], points2[triangle][1][1], points2[triangle][2][1])
            ymax = max(points2[triangle][0][1], points2[triangle][1][1], points2[triangle][2][1])
            return(xmin, xmax, ymin, ymax)

        #Find barycentric coordinates inside rectangle
        def barycentric_matrix_dest_inv(triangle):
            mat = [[points2[triangle][0][0], points2[triangle][1][0], points2[triangle][2][0]],
                   [points2[triangle][0][1], points2[triangle][1][1], points2[triangle][2][1]],
                   [1,1,1]]
            mat = np.asarray(mat)
            mat = np.linalg.inv(mat)
            return(mat)


        def barycentric_matrix_source(triangle):
            mat = [[points[triangle][0][0], points[triangle][1][0], points[triangle][2][0]],
                   [points[triangle][0][1], points[triangle][1][1], points[triangle][2][1]],
                   [1,1,1]]
            mat = np.asarray(mat)
            #print(mat.shape)
            return(mat)

        for m in range(tr1_index.shape[0]):
            
            xmin, xmax, ymin, ymax = bounding_rect(tr1_index[m])
            xmin2, xmax2, ymin2, ymax2 = bounding_rect2(tr1_index[m])
            bary_dest_inv = barycentric_matrix_dest_inv(tr1_index[m])
            bary_source = barycentric_matrix_source(tr1_index[m])
            
            
            #find barycentric coordinates of points inside the destination triangle 
            inside_triangle = []
            original_triangle = []
            for i in range(xmin2,xmax2):
                for j in range(ymin2,ymax2):
                    cart_dest = np.transpose([i, j, 1])
                    bary = np.dot(bary_dest_inv, cart_dest)
                    if ((0 <= bary[0] <= 1) and (0 <= bary[1] <= 1) and (0 <= bary[2] <= 1)):
                        inside_triangle.append(bary)
                        original_triangle.append([i,j,1])
            inside_triangle = np.asarray(inside_triangle)
            original_triangle = np.asarray(original_triangle, dtype = int)
            #find corresponding cartesian coordinates of matching points in source triangle
            correspond_triangle = np.zeros(inside_triangle.shape)
            
            for i in range(inside_triangle.shape[0]):
                correspond_triangle[i] = np.dot(bary_source, inside_triangle[i])
                correspond_triangle[i][0] = correspond_triangle[i][0]/correspond_triangle[i][2]
                correspond_triangle[i][1] = correspond_triangle[i][1]/correspond_triangle[i][2]
                correspond_triangle[i][2] = correspond_triangle[i][2]/correspond_triangle[i][2]
            
            correspond_triangle = np.asarray(correspond_triangle, dtype = int)
            
            #printing values of vertex coordinates of corresponding triangles in both images
            #print('test1', fake2[original_triangle[i][0]][original_triangle[i][1]])
            #print('test2', img[correspond_triangle[i][0]][correspond_triangle[i][1]])
            
            for i in range(inside_triangle.shape[0]):
                fake2[original_triangle[i][1]][original_triangle[i][0]] = img5[correspond_triangle[i][1]][correspond_triangle[i][0]]
                img6[original_triangle[i][1]][original_triangle[i][0]] = img5[correspond_triangle[i][1]][correspond_triangle[i][0]]

        #cv.imwrite('before_blend_2.jpg', img6)
        #cv.imwrite('after_blend_2.jpg', output)

        #find cutout center coordinate when it is blended with destination image
        cut_ymin = min(points2[:,1])
        cut_xmin = min(points2[:,0])
        cut_ymax = max(points2[:,1])
        cut_xmax = max(points2[:,0])

        center_y = int((cut_ymin+cut_ymax)/2)
        center_x = int((cut_xmin+cut_xmax)/2)

        fake2_gray = cv.cvtColor(fake2, cv.COLOR_BGR2GRAY)
        _,mask= cv.threshold(fake2_gray, 10, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)

        mask_width = cut_xmax-cut_xmin
        mask_height = cut_ymax - cut_ymin
        #fake2 = cv.bitwise_not(fake2)
        img6 = cv.bitwise_and(img6, img6, mask = mask_inv)
        output3 = cv.add(img6,fake2)
        output2 = cv.seamlessClone(output3, img7, mask, (center_x,center_y), cv.NORMAL_CLONE)
        count += 1
        output.write(output2)

# Driver Code
if __name__ == '__main__':

	# Calling the function
	FrameCapture("/Data/Data2.mp4")
