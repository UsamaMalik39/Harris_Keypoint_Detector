# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 23:13:33 2021

@author: Malik Usama
"""

import cv2 as cv2
import numpy as np
import math as math
import matplotlib.pyplot as plt
from PIL import Image as pimg




def obtainFCol0(img,rows,cols):
    result=np.zeros((rows,cols),dtype=int)
    for i in range (0,rows,1):
        for j in range (0,cols,1):
            if(j==0):
                result[i][j]=0
            else:
                result[i][j]=img[i][j-1]
    return result

def obtainFR0(img,rows,cols):
    result=np.array([0]*img)
    for i in range (0,rows,1):
        for j in range (0,cols,1):
            if(i==0):
                result[i][j]=0
            else:
                result[i][j]=img[i-1][j]
    return result

def elementWiseMul(arr1,arr2,rows,cols):
    result=np.zeros((rows,cols),dtype=int)
    for i in range(0,rows,1):
        for j in range(0,cols,1):
            result[i][j]=np.multiply(arr1[i][j],arr2[i][j])
    return result

def MeasureCArray(Ixsq,Iysq,IxIy,rows,cols,win_size):
    R=0
    C=0
    margin=int(np.floor(np.divide(win_size,2)))
    margin=2*margin
    M=np.zeros((2,2),dtype=int)
    k=0
    # delete right left top bottom row
    C_arr=np.zeros((rows-margin)*(cols-margin))
    while R<=rows-win_size:
        while C<=cols-win_size:
            for i in range(R,R+win_size-1,1):
                for j in range(C,C+win_size-1,1):
                    M[0][0]=M[0][0]+Ixsq[i][j]
                    M[0][1]=M[0][1]+IxIy[i][j]
                    M[1][0]=M[1][0]+IxIy[i][j]
                    M[1][1]=M[1][1]+Iysq[i][j]
            M=np.divide(M,np.power(win_size,2))
            detM=np.linalg.det(M)
            trace=np.trace(M)
            alpha=0.04
            C_arr[k]=np.subtract(detM,np.multiply(alpha,np.power(trace,2)))
            k=k+1
            M=np.zeros((2,2),dtype=int)
            C=C+1
        C=0
        R=R+1
    return C_arr


    
def thresholdCMatrix(givenMatrix,limit):
    size=givenMatrix.shape
    rows=size[0]
    cols=size[1]
    result=np.zeros((rows,cols))
    for i in range (0,rows,1):
        for j in range (0,cols,1):
            if (givenMatrix[i][j]<limit):
                result[i][j]=0
            else:
                result[i][j]=1
    return result
                
    
def findLocalMinCMatrix(myMat,win_size,myTHD):
    R=0
    C=0
    result_mat=myMat
    maxI=0
    maxJ=0
    size=myMat.shape
    rows=size[0]
    cols=size[1]
    to_fwd=np.zeros((rows,cols))
    while R<=rows-win_size+1:
        while C<=cols-win_size+1:
            maxI=R
            maxJ=C
            maxVal=myMat[maxI][maxJ]
            for i in range(R,R+win_size-1,1):
                for j in range(C,C+win_size-1,1):
                    if(maxVal<myMat[i][j]):
                        result_mat[maxI][maxJ]=0
                        
                        maxVal=myMat[i][j]
                        maxI=i
                        maxJ=j
                    else:
                        result_mat[i][j]=0
            to_fwd[maxI][maxJ]=result_mat[maxI][maxJ]
           
            if(  myTHD[maxI][maxJ]>0.0 ):
                result_mat[maxI][maxJ]=1
            C=C+1
        C=0
        R=R+1
    return result_mat,to_fwd
   

def makeRedDots(matRef,matR):
    size=matRef.shape
    rows=size[0]
    cols=size[1]
    result=matR
    for i in range (0,rows,1):
        for j in range (0,cols,1):
            if(matRef[i][j]==1):
                result[i][j][0]=0
                result[i][j][1]=0
                result[i][j][2]=255

    return result
#window for supreme court 7 and threshold 20000
#window for harry potter =7 and threshold 25000                
def runHaris(img,w_size,T):
    RGB_img=img
    img=cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)

    win_size=w_size
    """cv2.imshow('gray_orig',img)
    cv2.waitKey(0)
    
    cv2.imwrite("C:/EME/8th sem/CV/assignment/#1/pic/report/results/taskA/gray_hp.jpg",
                img)"""
    
    size=img.shape
    rows=size[0]
    cols=size[1]
    FC0=obtainFCol0(img, rows, cols)
    FR0=obtainFR0(img, rows, cols)  
    Ix=np.subtract(img,FC0)
    Iy=np.subtract(img,FR0)
    IxIy=elementWiseMul(Ix, Iy, rows, cols)
    Ixsq=elementWiseMul(Ix, Ix, rows, cols)
    Iysq=elementWiseMul(Iy, Iy, rows, cols)

    C_Array=MeasureCArray(Ixsq, Iysq, IxIy, rows, cols, win_size)
    margin=int(np.floor(np.divide(win_size,2)))
    margin=2*margin
    C_matrix=np.reshape(C_Array,(rows-margin,cols-margin))
    C_mat_THolded=thresholdCMatrix(C_matrix, T)
    """cv2.imshow('results-thresholded',C_mat_THolded)
    cv2.waitKey(0)"""
    
    C_mat_local_min,fwd_matrix=findLocalMinCMatrix(C_matrix, 
                                                   win_size,
                                                   C_mat_THolded)
    """cv2.imshow('results-with-local-min',C_mat_local_min)
    cv2.waitKey(0)"""
    
    
    Final_dotted_ver=makeRedDots(C_mat_local_min,RGB_img)
    cv2.imshow('final-dotted-overlapped',Final_dotted_ver)
    cv2.waitKey(0)
    cv2.imwrite("C:/EME/8th sem/CV/assignment/#1/pic/report/results/taskA/dotted_sc.jpg",
                Final_dotted_ver)
    
    return C_mat_local_min

##############################################################
def rotation_property(file_name,W_Z,T,orig_H_P):
    size=25
    M_array=np.zeros(size)
    N_array=0
    rep_array=np.zeros(size)
    j=0
    angles=np.zeros(size)
    for i in range (0,375,15):
        mimg=cv2.imread(file_name)
        Orig_Rot_H_P=rotate_img(orig_H_P,i)
        sec_Rot=rotate_img(mimg,i)
        cv2.imwrite("C:/EME/8th sem/CV/assignment/#1/pic/report/results/taskb/hp/image_.jpg",
                sec_Rot)
        sec_H_P=runHaris(sec_Rot, W_Z, T)
        
        M_array[j]=checkMatching(Orig_Rot_H_P,sec_H_P)
        print('matching checked: ', i)
        N_array=np.count_nonzero(Orig_Rot_H_P)
        
        rep_array[j]=M_array[j]/N_array
        
        angles[j]=i
        
        j=j+1
    return rep_array,angles

def checkMatching(orig,sec):
    M=0
    full_filled=sec
    iteration=np.count_nonzero(orig)
    non_zero_orig=np.transpose(np.nonzero(orig))
    x_val=np.zeros(iteration) # value of i
    y_val=np.zeros(iteration) # value of j

 
    for k in range(0,iteration,1):
        x_val[k]=non_zero_orig[k][0]
        y_val[k]=non_zero_orig[k][1]
        
        nearest_idx=nearest_nonzero_idx(full_filled, x_val[k], y_val[k])
        rot_x_val=nearest_idx[0]  # i value
        rot_y_val=nearest_idx[1]  # j value
        euc_dist=distance(x_val[k], y_val[k], rot_x_val, rot_y_val)
        if(euc_dist<5): # less than 5 units
            M=M+1
            print('found match: ',M)
            full_filled[rot_x_val][rot_y_val]=0
        
    return M
            
        
        
def nearest_nonzero_idx(a,x,y):
    idx = np.argwhere(a)
    print('nearest index returned')
    return idx[((idx - [x,y])**2).sum(1).argmin()]
    
        
            
        
        
    
def distance(x1,y1,x2,y2):
    dist=math.sqrt((x1-x2)**2+(y1-y2)**2)
    return dist

def rotate_img(img,angle):
    rows=img.shape[0]
    cols=img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def plot_graph(x_axis,y_axis,xlabel,ylabel):
    plt.plot(x_axis,y_axis,color='green',linestyle='solid',linewidth=2,
             marker='o',markerfacecolor='red',markersize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

#####################################################################

def scaling_property(fname,k_pts_orig,window,THD):
    size=9
    M_array=np.zeros(size)
    N_array=0
    rep_array=np.zeros(size)
    factors=np.zeros(size)
    for i in range(0,size,1):
        m_us_img=cv2.imread(fname)# rgb image
        s_fact=1.2**i
        factors[i]=s_fact
        if(s_fact==1):
            r_n_img=m_us_img  # resized but original     
        else:
            r_n_img=scaleImage(m_us_img, s_fact) #resized so new formed
        k_pts_resized=runHaris(r_n_img, window, THD)
        
        M_array[i]=checkMatching(k_pts_orig,k_pts_resized)
        print('matching checked: ', i)
        
        N_array=np.count_nonzero(k_pts_orig)
        
        rep_array[i]=M_array[i]/N_array
        
    return factors,rep_array
    
    
    
def scaleImage(image,fact):
    r=int(image.shape[0]*fact)
    c=int(image.shape[1]*fact)
    newDim=(r,c)
    resized = cv2.resize(image, newDim, interpolation = cv2.INTER_CUBIC)
    return resized


filename='C:/EME/8th sem/CV/assignment/#1/supreme_court.jpg'
mimg=cv2.imread(filename)


#harry_potters=3,500
#supreme_court=3,9000

key_pts=runHaris(mimg, 3, 9000)

"uncomment the next 2 line for task b"
#rot_rep,angles=rotation_property(filename, 3, 500, key_pts)
#plot_graph(angles,rot_rep,"angles","repeatability")


"uncomment the next  2 lines for task c"
#x_factors,y_rep=scaling_property(filename, key_pts, 3, 500)
#plot_graph(x_factors,y_rep,"factors","repeatability")

