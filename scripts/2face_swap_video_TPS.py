#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:55:23 2022

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

from numba import jit
import cv2 as cv
import dlib
import numpy as np

from timeit import default_timer as timer

def U(r):
    if(r == 0):
        return 0
    val = r**2
    ans = val * np.log(val)
    #print("r:",r," ,val:",val," ,ans:",ans)
    if(ans == np.nan):
        return 0
    return ans


# Function to extract frames
def FrameCapture(path):
    vidObj = cv.VideoCapture(path)
    width = int(vidObj.get(3))
    height = int(vidObj.get(4))
    frame_size = (width, height)
    fps = vidObj.get(5)
    output = cv.VideoWriter('Data/Data2OutputTPS.mp4',cv.VideoWriter_fourcc(*'MJPG'),30,frame_size)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    while success:
        success, img1 = vidObj.read()
        if not success:
            break
        scale_percent = 100 # percent of original size
        width = int(img1.shape[1] * scale_percent / 100)
        height = int(img1.shape[0] * scale_percent / 100)
        dim = (width, height)
          
        # resize image
        img1 = cv.resize(img1, dim, interpolation = cv.INTER_AREA)
        #img1 = cv.imread('Keshubh.jpeg')
        img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        img2 = img1
        scale_percent = 100 # percent of original size
        width = int(img2.shape[1] * scale_percent / 100)
        height = int(img2.shape[0] * scale_percent / 100)
        dim = (width, height)
          
        # resize image
        img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)

        #img2 = cv.imread('Dushyant1.jpg')
        img2_gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
        img3 = img1.copy()

        fake1 = np.zeros_like(img1)
        fake2 = np.zeros_like(img2)
        mask = np.zeros_like(img1_gray)
        #2. get face detector and landmark
        hog_face_detector = dlib.get_frontal_face_detector()
        landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        img6 = img2.copy()
        img5 = img1.copy()
        #3. get features of faces in image
        faces1 = hog_face_detector(img1_gray)
        #faces2 = hog_face_detector(img2_gray)
        landmarks1 = []
        landmarks2 = []
        face = faces1[0]
        face_landmarks = landmark(img1_gray,face)
        for n in range (68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            #cv.circle(img1, (x,y), 1, (255,0,0), 1)
            landmarks1.append([x,y])
        landmarks1 = np.asarray(landmarks1,dtype=np.float32)
        face2 = faces1[1]
        face_landmarks = landmark(img2_gray,face2)
        for n in range (68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            #cv.circle(img2, (x,y), 1, (255,0,0), 1)
            landmarks2.append([x,y])
        landmarks2 = np.asarray(landmarks2,dtype=np.float32)

        if (max(landmarks1[:,0]) <  min(landmarks2[:,0])):
            landmarks1 = np.asarray(landmarks1, np.int32)
            landmarks2 = np.asarray(landmarks2, np.int32)
            
        else:
            temp = np.asarray(landmarks1, np.int32)
            landmarks2 = temp
            landmarks1 = np.asarray(landmarks2, np.int32)

        convexhull = cv.convexHull(landmarks1)
        convexhull = np.asarray(convexhull, dtype = np.uint64)
        cv.fillConvexPoly(mask, convexhull, 255)


        #4. generate K
        K = np.zeros((68,68))
        for i in range (68):
            for j in range (68):
                if(i==j):
                    K[i][j] = 0
                    continue
                value = np.square(landmarks1[i][0] - landmarks1[j][0]) + np.square(landmarks1[i][1] - landmarks1[j][1])
                value = value**0.5
                value = U(value)
                if(value == 0 or value == np.nan):
                    #print(landmarks1[i][0], landmarks1[i][1], landmarks1[j][0], landmarks1[j][1])
                    K[i][j] = 0
                    continue
                K[i][j] = value
        print("K:\n",K.shape)  

        #5. generate P & PT
        P = np.zeros((68,3))
        for i in range (68):
            P[i][0] = 1
            P[i][1] = landmarks1[i][0]
            P[i][2] = landmarks1[i][1]
        PT = np.transpose(P)
        #print("P:\n",P.shape)
        #print("PT:\n",PT.shape)

        #6. generate identity matrix
        I = np.identity(71)

        #7. generate final matrx and inverse of it
        K_complex = np.zeros((71,71))
        for i in range (68):
            for j in range (68):
                K_complex[i][j] = K[i][j]
        for i in range (68):
            for j in range (3):
                K_complex[i][68 + j] = P[i][j]
        for i in range (3):
            for j in range (68):
                K_complex[68 + i][j] = PT[i][j]
        print("K_complex:\n",K_complex.shape)
        Lambda = 0.001
        finale = np.add(K_complex, Lambda*I)
        finale_inv = np.linalg.inv(finale)

        #8. find weights for x and y seperately
        v = np.zeros([K_complex.shape[0],2])
        for i in range(68):
            v[i] = landmarks2[i]
            
        #v = landmarks2

        dest_x = v[:,0]
        dest_y = v[:,1]

        coeff_x = np.dot(finale_inv,dest_x)
        coeff_y = np.dot(finale_inv,dest_y)

        a1_x = coeff_x[68]
        ax_x = coeff_x[69]
        ay_x = coeff_x[70]

        a1_y = coeff_y[68]
        ax_y = coeff_y[69]
        ay_y = coeff_y[70]

        #step 2 as per assignment notes
        # f(x,y) = a1 + a2x + a3y + sum(wi*U(xi,y-x,y))

        # uncomment below line to check if the size of dimension check matches the number of rows
        #dim_check = np.dot(K_complex[0:6,:],coeff_x)

        #create bounding box for face in destination
        face_ymin = int(round(min(landmarks1[:,1])))
        face_xmin = int(round(min(landmarks1[:,0])))
        face_ymax = int(round(max(landmarks1[:,1])))
        face_xmax = int(round(max(landmarks1[:,0])))

        center_y = int((face_ymin+face_ymax)/2)
        center_x = int((face_xmin+face_xmax)/2)

        #test tps function for face_xmin and face_ymin
        value1 = np.zeros([71,1])

        X_dash_points =np.zeros([((face_xmax-face_xmin)*(face_ymax-face_ymin)),4])
        x_value = np.zeros([71,1])
        y_value = np.zeros([71,1])

        x_dash = []
        y_dash = []
        orig_x = []
        orig_y = []
        flag = 0
        #tset
        for i in range(face_xmin,face_xmax):
            for j in range(face_ymin,face_ymax):
                a = 0
                val = np.square(landmarks1[0:68,0]-i)+np.square(landmarks1[0:68,1]-j)
                val = val + 1
                
                #val(a) = 0
                val = val*(np.log(val))
                for t in range(68):
                    x_value[t] = coeff_x[t]*val[t]
                    y_value[t] = coeff_y[t]*val[t]
                sumx = a1_x + ax_x*i + ay_x*j + sum(x_value)
                sumy = a1_y + ax_y*i + ay_y*j + sum(y_value)
                if sumx<0 :
                    print('val1')
                if sumy < 0:
                        print('val2')
                
                
                x_dash.append(sumx)
                y_dash.append(sumy)
                orig_x.append(i)
                orig_y.append(j)

        

        x_dash = np.asarray(x_dash, dtype = int)
        y_dash = np.asarray(y_dash, dtype = int)
        orig_x = np.asarray(orig_x, dtype = int)
        orig_y = np.asarray(orig_y, dtype = int)

        des_xmin = min(x_dash)
        des_xmax = max(x_dash)
        des_ymin = min(y_dash)
        des_ymax = max(y_dash)
        des_xmin = int(des_xmin)
        des_xmax = int(des_xmax)
        des_ymin = int(des_ymin)
        des_ymax = int(des_ymax)
        
        #cv.line(img2, (des_xmin,des_ymin), (des_xmin,des_ymax), [0,0,255], 2)
        #cv.line(img2, (des_xmin,des_ymin), (des_xmax,des_ymin), [0,0,255], 2)
        #cv.line(img2, (des_xmin,des_ymax), (des_xmax,des_ymax), [0,0,255], 2)
        #cv.line(img2, (des_xmax,des_ymin), (des_xmax,des_ymax), [0,0,255], 2)

        for i in range(x_dash.shape[0]):
            cv.circle(img6, (int(x_dash[i]), int(y_dash[i])), 2, [255,0,0], 2)
            img5[int(orig_y[i])][int(orig_x[i])] = img2[int(y_dash[i])][int(x_dash[i])]

        img5 = cv.bitwise_and(img5, img5, mask = mask)
        mask_inv = cv.bitwise_not(mask)
        img3 = cv.bitwise_and(img3,img3, mask = mask_inv)
        img3 = cv.add(img3,img5)
        img3 = cv.seamlessClone(img3, img1, mask, (center_x,center_y), cv.NORMAL_CLONE)
        count += 1
        output.write(img3)
        #cv.waitKey(0)

    vidObj.release()
    #cv.destroyAllWindows()

# Driver Code
if __name__ == '__main__':

	# Calling the function
	FrameCapture("Data/Data2.mp4")
