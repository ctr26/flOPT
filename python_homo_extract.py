# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2

theta = 0
bead_number = 6

bead_1 = np.transpose(np.matrix([0,0,0]))
bead_2 = np.transpose(np.matrix([0,0.50,0]))
bead_3 = np.transpose(np.matrix([0.50,0,0]))
bead_4 = np.transpose(np.matrix([0.50,0.25,0.50]))
bead_5 = np.transpose(np.matrix([-0.50,-0.50,-0.50]))
bead_6 = np.transpose(np.matrix([0,-0.50,-0.50]))
bead_7 = np.transpose(np.matrix([0,-0.50,-0.50]))

points_1 = np.empty((6,1,2), dtype=float)
points_2 = np.empty((6,1,2), dtype=float)

beads = np.matrix(np.random.rand(bead_number,1,3))
bead_pos = np.empty((bead_number,1,4))

point_1 = np.empty((bead_number,1,2), dtype=float)
point_2 = np.empty((bead_number,1,2), dtype=float)


#%%
angles = (np.linspace(0,2*np.pi,num=2)/100)

#for theta in angles:    
t_x = 0
t_y = 0
t_z = 0 

theta = angles[0]

rotation_matrix = np.matrix([[np.cos(theta)     ,np.sin(theta)  ,0, t_x],
                              [-np.sin(theta)   ,np.cos(theta)  ,0, t_y],
                              [0                ,0              ,1, t_z],
                              [0,0,0,1]
                              ])
for i in np.arange(bead_number):
    bead_pos[i] = (rotation_matrix*np.concatenate((beads[i].transpose(),np.matrix([1])),axis=0)).transpose()
    point_1[i] = bead_pos[i,0,0:2]/bead_pos[i,0,3]
#    
#bead_1_pos = rotation_matrix*np.concatenate((bead_1,np.matrix([1])),axis=0)
#bead_2_pos = rotation_matrix*np.concatenate((bead_2,np.matrix([1])),axis=0)
#bead_3_pos = rotation_matrix*np.concatenate((bead_3,np.matrix([1])),axis=0)
#bead_4_pos = rotation_matrix*np.concatenate((bead_4,np.matrix([1])),axis=0)
#bead_5_pos = rotation_matrix*np.concatenate((bead_5,np.matrix([1])),axis=0)
#bead_6_pos = rotation_matrix*np.concatenate((bead_6,np.matrix([1])),axis=0)
#
#'''points_1 = ((np.hstack((bead_1_pos[0:2],
#                      bead_2_pos[0:2],
#                      bead_3_pos[0:2],
#                      bead_4_pos[0:2]
#                      ))).transpose())'''
#points_1[0] = (bead_1_pos[0:2].transpose()).A
#points_1[1] = (bead_2_pos[0:2].transpose()).A
#points_1[2] = (bead_3_pos[0:2].transpose()).A
#points_1[3] = (bead_4_pos[0:2].transpose()).A
#points_1[4] = (bead_5_pos[0:2].transpose()).A
#points_1[5] = (bead_6_pos[0:2].transpose()).A
#

theta = angles[1]


rotation_matrix = np.matrix([[np.cos(theta)     ,np.sin(theta)  ,0, t_x],
                              [-np.sin(theta)   ,np.cos(theta)  ,0, t_y],
                              [0                ,0              ,1, t_z],
                              [0,0,0,1]
                              ])
    
for i in np.arange(bead_number):
    bead_pos[i] = (rotation_matrix*np.concatenate((beads[i].transpose(),np.matrix([1])),axis=0)).transpose()
    point_2[i] = bead_pos[i,0,0:2]/bead_pos[i,0,3]
    
    
#bead_1_pos = rotation_matrix*np.concatenate((bead_1,np.matrix([1])),axis=0)
#bead_2_pos = rotation_matrix*np.concatenate((bead_2,np.matrix([1])),axis=0)
#bead_3_pos = rotation_matrix*np.concatenate((bead_3,np.matrix([1])),axis=0)
#bead_4_pos = rotation_matrix*np.concatenate((bead_4,np.matrix([1])),axis=0)
#bead_5_pos = rotation_matrix*np.concatenate((bead_5,np.matrix([1])),axis=0)
#bead_6_pos = rotation_matrix*np.concatenate((bead_6,np.matrix([1])),axis=0)
#
#points_2[0] = (bead_1_pos[0:2].transpose()).A
#points_2[1] = (bead_2_pos[0:2].transpose()).A
#points_2[2] = (bead_3_pos[0:2].transpose()).A
#points_2[3] = (bead_4_pos[0:2].transpose()).A
#points_1[4] = (bead_5_pos[0:2].transpose()).A
#points_1[5] = (bead_6_pos[0:2].transpose()).A
    
#camera_matrix = np.matrix('50,0,25;0,50,25;0,0,1')

camera_matrix = np.matrix('1,0,0;0,1,0;0,0,1')
E, mask = cv2.findEssentialMat(point_2,point_1)
R1,R2,t = cv2.decomposeEssentialMat(E)
#points, R, t, mask = cv2.recoverPose(E, points1, points2)
    
#E = findEssentialMat(points1, points2, focal, pp, RANSAC, 0.999, 1.0, mask);
#A = recoverPose(E, points1, points2, R, t, focal, pp, mask);
    
print((rotation_matrix[0:3,0:3]-R1))
print((rotation_matrix[0:3,0:3]-R2))
print(t)
        