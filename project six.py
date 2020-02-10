# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:06:26 2019

@author: Mohammadmahdi
"""
import numpy as np 
import cv2 as cv 
import needfcn as nf
import math as mt
import matplotlib.pyplot as plt 


img_name = ['MAP.tif','ML.tif','MD.tif','SAM.tif',
            'CC.tif','MF.tif','PP.tif','NN.tif']

noc = 8     # number of classes
noC = 5     # number of classifiers

clst = np.zeros((80,120,8))     # classified image storage

for i in range(8):
    clst[:,:,i] = cv.imread(img_name[i],0)
    
m,n = np.shape(clst[:,:,1])
clst_new = np.delete(clst,[1,2,4],axis=2)
    
#%% data cordinate 

interval1,interval2,interval = nf.void()

#%%   Decision making
  
dfmmfe = np.zeros((m,n,3))

for i in range(m):
    for j in range(n):
        dfmmfe[i,j,0] = nf.major(clst_new[i,j,:])
        dfmmfe[i,j,1] = nf.fifper(clst_new[i,j,:])
        dfmmfe[i,j,2] = nf.eyper(clst_new[i,j,:])
        
plt.figure()        
plt.imshow(dfmmfe[:,:,0],cmap="gray"),plt.title('majarity')
plt.xticks([]),plt.yticks([])
plt.figure()      
plt.imshow(dfmmfe[:,:,1],cmap="gray"),plt.title('fifty percent voting decision maker classifier')
plt.xticks([]),plt.yticks([])
plt.figure()      
plt.imshow(dfmmfe[:,:,2],cmap="gray"),plt.title('eighty percent voting decision maker classifier')
plt.xticks([]),plt.yticks([])

filename = ['majority.tif','50.tif','80.tif']



#%% confusion matrix of all classifiers

noC = 3    
confusion_matrix = np.zeros((noc+1,noc+1,noC))

for k in range(noC):
    for c in range(noc):
        if (c==0):
            for t in range(0,8,2): 
                for i in interval1[t]:
                    for j in interval1[t+1]: 

                        confusion_matrix[:noc,:noc,k] = nf.counter(dfmmfe[i,j,k],confusion_matrix[:noc,:noc,k],c)         
        elif (c==1):
            for t in range(0,8,2): 
                for i in interval2[t]:
                    for j in interval2[t+1]: 
                        confusion_matrix[:noc,:noc,k] = nf.counter(dfmmfe[i,j,k],confusion_matrix[:noc,:noc,k],c)
        else:
            for i in interval[c-2][0]:
                for j in interval[c-2][1]:
                    confusion_matrix[:noc,:noc,k] = nf.counter(dfmmfe[i,j,k],confusion_matrix[:noc,:noc,k],c)
       

#%% OA OV 
    
for j in range(noC):    
    for i in range(noc):
        confusion_matrix[i,noc,j] = round(100*(confusion_matrix[i,i,j]/np.sum(confusion_matrix[i,:,j],axis=0)),2)
        confusion_matrix[noc,i,j] = round(100*(confusion_matrix[i,i,j]/np.sum(confusion_matrix[:,i,j],axis=0)),2)
        if (mt.isnan(float(confusion_matrix[noc,i,j]))):
            confusion_matrix[noc,i,j]=0


#%% overall accuracy and average validity 


PCi = np.array([[0.25,0.25,1/12,1/12,1/12,1/12,1/12,1/12]]).transpose()     
OA = np.zeros((1,noC))
AV = np.zeros((1,noC))
AA = np.zeros((1,noC))
OV = np.zeros((1,noC))

for i in range(noC):
    OA[0,i] = np.round(100*np.sum(np.diag(confusion_matrix[0:noc,0:noc,i],k=0))/np.sum(np.sum(confusion_matrix[0:noc,0:noc,i])))
    AV[0,i] = np.round((1/noc)*np.sum(confusion_matrix[noc,:,i]))
    AA[0,i] = np.round((1/noc)*np.sum(confusion_matrix[:,noc,i]))


OA_new = OA/100

Ni = np.sum(confusion_matrix[:noc,:noc,:],axis = 1)
Mj = np.sum(confusion_matrix[:noc,:noc,:],axis = 0)
N = 9600
kappa = np.zeros((1,noC))
Pe = np.zeros((1,noC))
for i in range(noC):
    Pe[0,i] = np.matmul(np.transpose(Ni[:,i]),Mj[:,i])/(9600**2)
    kappa[0,i] = (OA_new[0,i] - Pe[0,i])/(1 - Pe[0,i])


#%% unlabeled sampled 

counter_mat = np.zeros((9,10,3))                    
counter_mat[0:9,1:10,:] = confusion_matrix

for k in range(3):
    for i in range(m):
        for j in range(n):
            if (dfmmfe[i,j,k]==255):
                counter_mat[8,0,k] = counter_mat[8,0,k] + 1
                    

#%% plotting the data
        
import seaborn as sb 

name1 = ['Majority','Fifty percent','Eighty percent']
name2 = ['C1','C2','C3','C4','C5','C6','C7','C8','Acc']  
name3 = ['C1','C2','C3','C4','C5','C6','C7','C8','Val'] 
   
for i in range(noC):
    plt.figure(i+1),sb.heatmap(confusion_matrix[:,:,i], xticklabels=name2, yticklabels=name3,annot=True,annot_kws={"size": 10},fmt='.0f')    
    plt.title(name1[i])


#%% OV
    
for i in range(noC):
    OV[0,i] = np.round(np.matmul(confusion_matrix[noc,:noc,i],PCi))
    