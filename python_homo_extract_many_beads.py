import numpy as np
import cv2

np.random.seed(1)

theta = 0
bead_number = 6

beads = (np.matrix(np.random.rand(bead_number,1,3))-0.5)
bead_pos = np.empty((bead_number,1,4))

point_1 = np.empty((bead_number,1,2), dtype=float)
point_2 = np.empty((bead_number,1,2), dtype=float)


#%%
angles = (np.linspace(0,2*np.pi,num=2)/50)

#for theta in angles:    
t_x = 0
t_y = 0
t_z = 0 

theta = angles[0]

rotation_matrix = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
for i in np.arange(bead_number):
    bead_pos[i] = (rotation_matrix*np.concatenate((beads[i].transpose(),np.matrix([1])),axis=0)).transpose()
    point_1[i] = bead_pos[i,0,0:2]/bead_pos[i,0,3]

theta = angles[1]

t_x = 0
t_y = 1
t_z = 0 

rotation_matrix = np.matrix([[np.cos(theta)     ,np.sin(theta)  ,0, t_x],
                              [-np.sin(theta)   ,np.cos(theta)  ,0, t_y],
                              [0                ,0              ,1, t_z],
                              [0,0,0,1]
                              ])

#rotation_matrix = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    
for i in np.arange(bead_number):
    bead_pos[i] = (rotation_matrix*np.concatenate((beads[i].transpose(),np.matrix([1])),axis=0)).transpose()
    point_2[i] = bead_pos[i,0,0:2]/bead_pos[i,0,3]
    

    
#camera_matrix = np.matrix('50,0,25;0,50,25;0,0,1')

#camera_matrix = np.matrix('20,0,100;0,20,100;0,0,1')
#
#E, mask = cv2.findEssentialMat(point_1,point_2)
#R1,R2,t = cv2.decomposeEssentialMat(E)
#points, R, t, mask = cv2.recoverPose(E, point_1, point_2)

K = np.matrix('1,0,0;0,1,0;0,0,1')

H,inliers = cv2.findHomography(point_1,point_2)
a,R,T,translation = cv2.decomposeHomographyMat(H,K)
    
#E = findEssentialMat(points1, points2, focal, pp, RANSAC, 0.999, 1.0, mask);
#A = recoverPose(E, points1, points2, R, t, focal, pp, mask);



        