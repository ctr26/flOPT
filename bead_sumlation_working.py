#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:06:21 2017

@author: craggles
"""
import numpy as np
import cv2
from drawnow import drawnow, figure

from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt

lena = cv2.imread('lena-128x128.jpg')
lena = cv2.cvtColor(lena,cv2.COLOR_RGB2GRAY) # RGBtoGray
lena = cv2.normalize(lena.astype('float'),  None, 0, 1, cv2.NORM_MINMAX) #mat2gray

plt.gray()
#plt.imshow(lena)

flag_record = 1;
width = 128;
image = np.zeros((width,width));
image_fusion =np.zeros((width*2,width*2));
radius = 10;
beads = 4

fov = np.arange(-width/2,width/2,1)

[x,y,z] = np.meshgrid(fov,fov,fov)

#r = np.sqrt(x**2 + y**2)

bead = [[]]*beads
bead_pos = [[]]*beads

bead[0] = np.array([0,0,-50])#) bead_1_pos = [];
bead[1] = np.array([0,50,0])# bead_2_pos = [];
bead[2] = np.array([-50,0,0]) #bead_3_pos = [];
bead[3] = np.array([0,0,-50])# bead_4_pos = [];

r = [[]]*len(bead)

for i in np.arange(len(bead)):
    r[i] = np.sqrt((x-bead[i][0])**2 + (y-bead[i][1])**2 + (z-bead[i][2])**2)
    
bead_volume = ((r[0]<(radius)) | (r[1]<radius) | (r[2]<radius) | (r[3]<radius)).astype(float)

plt.subplot(2,2,1)
plt.imshow(bead_volume[:,:,round(width/2)])

bead_volume[:,:,round(width/2)] = np.maximum(bead_volume[:,:,round(width/2)],lena)
#
#for i in np.arange(width):
plt.imshow(bead_volume[:,:,round(width/2)])
#    drawnow

angles = np.linspace(0,2*np.pi,round(width/2))
#angles = np.linspace(0,0.1*np.pi,2)

reconstruction_back_projection = np.empty((width,width))

for idx,theta in enumerate(angles):
    
#    print(theta)
    print(idx)
    
    t_x = 0;
    t_y = 0;
    t_z = 0;
    rotation_matrix = np.matrix(np.array([[np.cos(theta)      ,np.sin(theta)  ,0  ,t_x],
                                 [-np.sin(theta)    ,np.cos(theta)  ,0  ,t_y],
                                 [0                 ,0              ,1  ,t_z],
                                 [0                 ,0              ,0  ,1]
                                 ]))
    
    centre=0.5*np.array(bead_volume.shape)
    rot = rotation_matrix[0:3,0:3]
    offset=np.array((centre-centre.dot(rot)).dot(np.linalg.inv(rot)))
    dest_shape = (width*2, width*2,width*2)
    ' Homogenous transform'
    #rotation_matrix_corrected = rotation_matrix
    #rotation_matrix_corrected[0:3,3] = rotation_matrix_corrected[0:3,3] + (np.matrix((width/2,width/2,width/2))).T
    transformed_volume_no_t = ndimage.interpolation.affine_transform(bead_volume,rot,order=2,offset=-((offset.T).flatten()))
    #Fix lack of trnaslation
    plt.subplot(2,2,2)
    plt.imshow(transformed_volume_no_t[:,:,round(width/2)])
    ' Calculate homogenous new coordinates'
    for j,element in enumerate(bead):
        bead_pos[j] = rotation_matrix*np.concatenate((np.matrix(bead[j]).T,(np.matrix(1)).T))
    ' Projection'
    projection = np.sum(transformed_volume_no_t,axis=0) #Probably axis 2?
    back_projection = np.tile(projection,(width,1,1)) #check extensively.
    #transformed_back_projection = ndimage.interpolation.affine_transform(back_projection,(rotation_matrix.I))
    offset=np.array((centre-centre.dot(rot.I)).dot(np.linalg.inv(rot.I)))
    transformed_back_projection = ndimage.interpolation.affine_transform(back_projection,(rot.I),order=2,offset=-((offset.T).flatten()))
    reconstruction_back_projection = transformed_back_projection + reconstruction_back_projection

plt.subplot(2,2,3)
plt.imshow(reconstruction_back_projection[:,:,round(width/2)])    
    

    