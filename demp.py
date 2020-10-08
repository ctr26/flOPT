#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:05:49 2017

@author: craggles
"""
import numpy as np
import cv2

point_1 = (np.random.rand(100,1,2))

E, mask = cv2.findEssentialMat(point_1,point_1+1)
R1,R2,t = cv2.decomposeEssentialMat(E)
points, R_pose, t_pose, mask = cv2.recoverPose(E, point_1, point_1+1)