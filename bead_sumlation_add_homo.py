#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:06:21 2017

@author: craggles
"""
import numpy as np
import sys
import cv2
from scipy import misc
from scipy import ndimage
#import matplotlib.pyplot as plt

import pylab as plt
from drawnow import drawnow, figure
#%% Plotting

fig = plt.figure()

ax0 = fig.add_subplot(221) 
ax1 = fig.add_subplot(222) 
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
#%% Import Lena

lena = cv2.imread('lena-128x128.jpg')
lena = cv2.cvtColor(lena,cv2.COLOR_RGB2GRAY) # RGBtoGray
lena = cv2.normalize(lena.astype('float'),  None, 0, 1, cv2.NORM_MINMAX) #mat2gray
plt.gray()
plt.subplot(2,2,1);plt.subplot(2,2,2);plt.subplot(2,2,3);plt.subplot(2,2,4)
plt.tight_layout(pad=1.2)
#plt.imshow(lena)

flag_record = 1;
width = 128;
image = np.zeros((width,width));
image_fusion =np.zeros((width*2,width*2))
bead_volume_bool = np.empty((width,width,width),bool)
radius = 6;
beads = 6

fov = np.arange(-width/2,width/2,1)

[x,y,z] = np.meshgrid(fov,fov,fov)

#r = np.sqrt(x**2 + y**2)

bead = [[]]*beads
bead_pos = [[]]*beads
bead = np.empty((beads,3))

bead[0] = np.array([0,0,0])#) bead_1_pos = [];
bead[1] = np.array([0,50,0])# bead_2_pos = [];
bead[2] = np.array([50,0,0]) #bead_3_pos = [];
bead[3] = np.array([50,-50,0])# bead_4_pos = [];
bead[4] = np.array([-50,0,0]) #bead_3_pos = [];
bead[5] = np.array([-50,50,0])# bead_4_pos = [];

r = [[]]*len(bead)

for i in np.arange(len(bead)):
    bead_volume_bool = (bead_volume_bool) | ((np.sqrt((x-bead[i][0])**2 + (y-bead[i][1])**2 + (z-bead[i][2])**2))<(radius)) #Volume image of each bead
    

#bead_volume = ((r[0]<(radius)) | (r[1]<radius) | (r[2]<radius) | (r[3]<radius)).astype(float) #Combine all

bead_volume = bead_volume_bool.astype(float)

#%% Plot Image
#plt.subplot(2,2,1)
#plt.imshow(bead_volume[:,:,round(width/2)])
bead_volume[:,:,round(width/2)] = np.maximum(bead_volume[:,:,round(width/2)],lena)
#plt.imshow(bead_volume[:,:,round(width/2)])
#    drawnow
angles = np.linspace(0,2*np.pi,round(width/4))
#angles = np.linspace(0,0.1*np.pi,2)
reconstruction_back_projection = np.empty((width,width))
transformed_volume = np.empty((width,width))
projection = np.empty((width,width))

bead_pos = np.empty((beads,angles.size,4))
bead_pos_new = np.empty((beads,angles.size,4))
sinugram = np.empty((128,len(angles)))

#%% Plotting space setup
angle_idx=0

def draw_fig():
    plt.subplot(2,2,1)
    plt.imshow(bead_volume[:,:,round(width/2)])
    plt.title('Original')
    
    plt.subplot(2,2,2)
    plt.imshow(transformed_volume[:,:,round(width/2)])
    plt.title('Slice (xy)')
        
    plt.subplot(2,2,3)
    plt.imshow(projection)
    plt.title('Projection (xz)')
    
    plt.subplot(2,2,4)
    plt.imshow(reconstruction_back_projection[:,:,round(width/2)])
    plt.title('Reconstruction (xy)')

    
    plt.savefig('im/drift_beads_homo_test3_'+str(angle_idx))
    #show()
    
#%%
for angle_idx,theta in enumerate(angles):

#    print(theta)
    print(angle_idx)

    t_x = theta*2;
    t_y = theta*2;
    t_z = 0;
    rotation_matrix = np.matrix(np.array([[np.cos(theta)    ,np.sin(theta)  ,0  ,t_x],
                                           [-np.sin(theta)  ,np.cos(theta)  ,0  ,t_y],
                                           [0               ,0              ,1  ,t_z],
                                           [0               ,0              ,0  ,1]
                                           ]))

    centre=0.5*np.array(bead_volume.shape)
    rot = rotation_matrix[0:3,0:3]
    trans = rotation_matrix[0:3,3]
    offset=np.array((centre-centre.dot(rot)).dot(np.linalg.inv(rot)))
    #offset=np.array((t_x,t_y,t_z))+np.array((centre-centre.dot(rot)).dot(np.linalg.inv(rot))) Adding translation vector onto intrinsic offset,
    dest_shape = (width*2, width*2,width*2)
    ' Homogenous transform'
    #rotation_matrix_corrected = rotation_matrix
    #rotation_matrix_corrected[0:3,3] = rotation_matrix_corrected[0:3,3] + (np.matrix((width/2,width/2,width/2))).T
    transformed_volume_no_t = ndimage.interpolation.affine_transform(bead_volume,rot,
                                                                     offset=-((offset.T).flatten()))
    transformed_volume = ndimage.interpolation.shift(transformed_volume_no_t,trans)
    #Fix lack of trnaslation
#    plt.clf()
#    plt.subplot(2,2,2)
#    plt.imshow(transformed_volume[:,:,round(width/2)])
    ' Calculate homogenous new coordinates'
    for j,element in enumerate(bead):
        bead_pos[j,angle_idx,:] = (rotation_matrix*np.concatenate((np.matrix(bead[j]).T,(np.matrix(1)).T))).flatten()
    #%%
    first_xy = bead[:,0:2]/width
    current_xy = bead_pos[:,angle_idx,0:2]/width
#    if angle_idx == 0:
#        current_xy = bead_pos[:,angle_idx,0:2]/width
#        previous_xy = bead[:,0:2]/width
#        
#    else:
    
#        previous_xy = bead_pos[:,angle_idx-1,0:2]/width
    ' Find E'
    K = np.matrix('1,0,0;0,1,0;0,0,1')
    E,mask = cv2.findEssentialMat(first_xy,current_xy)
    R1,R2,t = cv2.decomposeEssentialMat(E)
    points, R_pose, t_pose, mask = cv2.recoverPose(E,first_xy,current_xy)
    ' Find H'
    H,inliers = cv2.findHomography(first_xy,current_xy)
    a,R,T,translation = cv2.decomposeHomographyMat(H,K)   
    T = np.multiply(T,width)
#    for j,element in enumerate(bead):     
#        bead_pos_new[j,angle_idx,:] = ((R.I)*np.concatenate((np.matrix(bead[j]).T,(np.matrix(1)).T))).flatten()
    #%% Cheating step to find which mmatrices are right
    square_difference_R = np.empty(len(R))
    square_difference_T = np.empty(len(T))
    for count,current_R in enumerate(R):
        square_difference_R[count] = sum(sum(((np.array(rot)) - (current_R))**2))
        
    for count,current_T in enumerate(T):
        square_difference_T[count] = sum(sum(((np.array(trans)) - (current_T))**2))
        
    #%% Update rotation matrix
    rot = np.matrix(R[np.argmin(square_difference_R)])
    trans = np.matrix(T[np.argmin(square_difference_T)])    
#    for j,element in enumerate(bead):     
#        bead_pos_new[j,angle_idx,:] = np.concatenate([np.array(((rot.I)*np.matrix(bead[j]).T)).flatten(),np.array([1])])
    #%% Projection
    ' Projection'
    projection = np.sum(transformed_volume,axis=0) #Probably axis 2?
    sinugram[:,angle_idx] = np.sum(projection,axis=1)
    #%%
    back_projection = np.tile(projection,(width,1,1)) #check extensively.
    #transformed_back_projection = ndimage.interpolation.affine_transform(back_projection,(rotation_matrix.I))
    back_projection = ndimage.interpolation.shift(back_projection,-trans)
    offset=np.array((centre-centre.dot(rot.I)).dot(np.linalg.inv(rot.I)))
    transformed_back_projection = ndimage.interpolation.affine_transform(back_projection,(rot.I),
                                                                         order=2,offset=-((offset.T).flatten()))
    reconstruction_back_projection = transformed_back_projection + reconstruction_back_projection
    drawnow(draw_fig,angle_idx)
    

#plt.subplot(2,2,4)
#plt.imshow(reconstruction_back_projection[:,:,round(width/2)])
#
#ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
#plt.show()
