# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:40:42 2019

@author: Mohammadmahdi
"""

def mapper(inp):
    if (inp==0):
        inp = 32
    elif (inp==1):
        inp = 64
    elif (inp==2):
        inp = 96
    elif (inp==3):
        inp = 128
    elif (inp==4):
        inp = 159
    elif (inp==5):
        inp = 191
    elif (inp==6):
        inp = 223
    elif (inp==7):
        inp = 0  
    return inp

def counter(condition,confusion,c):
    if (condition==32):
        confusion[c,0] = confusion[c,0] + 1
    elif (condition==64):
        confusion[c,1] = confusion[c,1] + 1
    elif (condition==96):
        confusion[c,2] = confusion[c,2] + 1
    elif (condition==128):
        confusion[c,3] = confusion[c,3] + 1
    elif (condition==159):
        confusion[c,4] = confusion[c,4] + 1
    elif (condition==191):
        confusion[c,5] = confusion[c,5] + 1
    elif (condition==223):
        confusion[c,6] = confusion[c,6] + 1
    elif (condition==0):
        confusion[c,7] = confusion[c,7] + 1
    return confusion 


def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom') 
import numpy as np 

def major(inp):
    m = np.shape(inp)
    ind = np.zeros((256,))
    for i in range(m[0]):
        ind[int(inp[i])] = ind[int(inp[i])] + 1
    a = np.argmax(ind)
    return a

def fifper(inp):
    # fifty percent voting decision maker 
    m = np.shape(inp)
    ind = np.zeros((256,))
    for i in range(m[0]):
        ind[int(inp[i])] = ind[int(inp[i])] + 1
    a=5
    
    for i in range(256):
        if (ind[i]>=2):
            a = i
    if (a==5):
        a=255
    return a
            
def eyper(inp):
    # eighty percent voting decision maker 
    m = np.shape(inp)
    ind = np.zeros((256,))
    for i in range(m[0]):
        ind[int(inp[i])] = ind[int(inp[i])] + 1
    a=5
    
    for i in range(256):
        if (ind[i]>=3):
            a = i
    if (a==5):
        a=255
    return a    
                
                
                
                
                
                
                                    