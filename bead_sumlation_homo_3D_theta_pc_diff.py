# %%
import seaborn as sns
import pandas as pd
from drawnow import drawnow, figure
import pylab as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy import ndimage
from scipy import misc
from scipy import io
import cv2
import sys
import numpy as np
import copy

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:06:21 2017

@author: Craig Russell
"""
# import matplotlib.pyplot as plt

# %% Plotting

fig = plt.figure()

ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)

## Import Lena

lena = cv2.imread("./lena-128x128.jpg")
lena = cv2.cvtColor(lena, cv2.COLOR_RGB2GRAY)  # RGBtoGray
lena = cv2.normalize(lena.astype("float"), None, 0, 1, cv2.NORM_MINMAX)  # mat2gray

plt.gray()
plt.subplot(2, 2, 1)
plt.subplot(2, 2, 2)
plt.subplot(2, 2, 3)
plt.subplot(2, 2, 4)
plt.tight_layout(pad=1.2)
# plt.imshow(lena)

flag_record = 1
width = 128
image = np.zeros((width, width))
image_fusion = np.zeros((width * 2, width * 2))
bead_volume_bool = np.empty((width, width, width), bool)
radius = 6
beads = 8


def metadata(suffix):
    import datetime

    date = datetime.datetime.today().strftime("%m%d%Y")
    return f"./graphs/{str(date)}_bead_simulation_{suffix}.pdf"


fov = np.arange(-width / 2, width / 2, 1)

[x, y, z] = np.meshgrid(fov, fov, fov)

# r = np.sqrt(x**2 + y**2)

bead = [[]] * beads
bead_pos = [[]] * beads
bead = np.empty((beads, 3))

bead[0] = np.array([0, 0, 10])  # ) bead_1_pos = [];
bead[1] = np.array([0, 50, 10])  # bead_2_pos = [];
bead[2] = np.array([50, 0, 10])  # bead_3_pos = [];
bead[3] = np.array([50, -50, 10])  # bead_4_pos = [];
bead[4] = np.array([-50, 0, 10])  # bead_3_pos = [];
bead[5] = np.array([-50, 50, 10])  # bead_4_pos = [];
bead[6] = np.array([-25, 0, 10])  # bead_3_pos = [];
bead[7] = np.array([-25, 25, 10])  # bead_4_pos = [];

bead = (np.random.rand(beads, 3) - 0.5) * 128
r = [[]] * len(bead)

for i in np.arange(len(bead)):
    bead_volume_bool = (bead_volume_bool) | (
        (np.sqrt((x - bead[i][0]) ** 2 + (y - bead[i][1]) ** 2 + (z - bead[i][2]) ** 2))
        < (radius)
    )  # Volume image of each bead


# bead_volume = ((r[0]<(radius)) | (r[1]<radius) | (r[2]<radius) | (r[3]<radius)).astype(float) #Combine all

bead_volume = bead_volume_bool.astype(float)

# %% Plot Image
# plt.subplot(2,2,1)
# plt.imshow(bead_volume[:,:,round(width/2)])
bead_volume[:, :, round(width / 2)] = np.maximum(
    bead_volume[:, :, round(width / 2)], lena
)
# plt.imshow(bead_volume[:,:,round(width/2)])
#    drawnow
angles = np.linspace(0, 2 * np.pi, round(width / 4))

# angles = np.linspace(0,0.1*np.pi,2)
reconstruction_back_projection = np.empty((width, width))
transformed_volume = np.empty((width, width))
projection = np.empty((width, width))

bead_pos = np.empty((beads, angles.size, 4))
bead_pos_new = np.empty((beads, angles.size, 4))
sinugram = np.empty((128, len(angles)))
unit_pose_scaling = np.empty((len(angles), 3))

# %% Plotting space setup
angle_idx = 0

# def draw_fig():
#    plt.subplot(2,2,1)
#    plt.imshow(bead_volume[:,:,round(width/2)])
#    plt.title('Original')
#
#    plt.subplot(2,2,2)
#    plt.imshow(transformed_volume[:,:,round(width/2)])
#    plt.title('Slice (xy)')
#
#    plt.subplot(2,2,3)
#    plt.imshow(projection)
#    plt.title('Projection (xz)')
#
#    plt.subplot(2,2,4)
#    plt.imshow(reconstruction_back_projection[:,:,round(width/2)])
#    plt.title('Reconstruction (xy)')
#
#
#    plt.savefig('im/drift_beads_homo_noncoplanar_rand'+str(angle_idx))
#    #show()
# %%
helical_shifts = np.linspace(0, 50, 20)
pc_sum_trans = np.empty((angles.size, 20))
pc_sum_rot = np.empty((angles.size, 20))


drift = np.linspace(0, 1, 20)
alpha_drift = drift * 0.5
translation_drift = drift * 50
list_of_dfs = {}


# t_x_helix = np.linspace(0, helicity, angles.size)
# alpha_helix = t_x_helix/100
# beta_helix = t_x_helix/100
conditions = {
    "Tx": {
        "t_x_helix": translation_drift,
        "alpha_helix": drift * 0,
        "beta_helix": drift * 0,
        "helical_shifts": translation_drift,
    },
    "alpha": {
        "t_x_helix": alpha_drift,
        "alpha_helix": drift * 0,
        "beta_helix": drift * 0,
        "helical_shifts": alpha_drift,
    },
}
for condition_name in conditions:
    print(condition_name)
    condition = conditions[condition_name]
    helical_shifts = condition["helical_shifts"]
    for helicity_idx, helicity in enumerate(helical_shifts):
        for angle_idx, theta in enumerate(angles):
            # condition_name = condition["condition_name"]
            t_x_helix = condition["t_x_helix"]
            alpha_helix = condition["alpha_helix"]
            beta_helix = condition["beta_helix"]

            #    print(theta)
            print(angle_idx)

            t_x = t_x_helix[helicity_idx]  # ;theta*2
            t_y = 0  # theta*2
            t_z = 0

            # for helicity_idx, helicity in enumerate(helical_shifts):

            #     t_x_helix = np.linspace(0, helicity, angles.size)
            #     alpha_helix = t_x_helix/100
            #     beta_helix = t_x_helix/100

            #     for angle_idx, theta in enumerate(angles):

            #         #    print(theta)
            #         print(angle_idx)

            #         t_x = t_x_helix[angle_idx]  # ;theta*2
            #         t_y = 0  # theta*2
            #         t_z = 0

            #        rotation_matrix = np.matrix(np.array([[np.cos(theta)    ,np.sin(theta)  ,0  ,t_x(angle_idx)],
            #                                               [-np.sin(theta)  ,np.cos(theta)  ,0  ,t_y],
            #                                               [0               ,0              ,1  ,t_z],
            #                                               [0               ,0              ,0  ,1]
            #                                               ]))
            alpha = alpha_helix[helicity_idx]  # 0#theta/100
            beta = 0  # beta_helix[angle_idx]
            gamma = theta

            rotation_matrix = np.matrix(
                np.array(
                    [
                        [
                            np.cos(beta) * np.cos(gamma),
                            np.cos(beta) * np.sin(gamma),
                            -np.sin(beta),
                            t_x,
                        ],
                        [
                            np.sin(alpha) * np.sin(beta) * np.cos(gamma)
                            - np.cos(alpha) * np.sin(gamma),
                            np.sin(alpha) * np.sin(beta) * np.sin(gamma)
                            + np.cos(alpha) * np.cos(gamma),
                            np.sin(alpha) * np.cos(beta),
                            t_y,
                        ],
                        [
                            np.cos(alpha) * np.sin(beta) * np.cos(gamma)
                            + np.sin(alpha) * np.sin(gamma),
                            np.cos(alpha) * np.sin(beta) * np.sin(gamma)
                            - np.sin(alpha) * np.cos(gamma),
                            np.cos(alpha) * np.cos(beta),
                            t_z,
                        ],
                        [0, 0, 0, 1],
                    ]
                )
            )

            centre = 0.5 * np.array(bead_volume.shape)
            rot = rotation_matrix[0:3, 0:3]
            trans = rotation_matrix[0:3, 3]
            offset = np.array((centre - centre.dot(rot)).dot(np.linalg.inv(rot)))
            # offset=np.array((t_x,t_y,t_z))+np.array((centre-centre.dot(rot)).dot(np.linalg.inv(rot))) Adding translation vector onto intrinsic offset,
            dest_shape = (width * 2, width * 2, width * 2)
            " Homogenous transform"
            # rotation_matrix_corrected = rotation_matrix
            # rotation_matrix_corrected[0:3,3] = rotation_matrix_corrected[0:3,3] + (np.matrix((width/2,width/2,width/2))).T
            #    transformed_volume_no_t = ndimage.interpolation.affine_transform(bead_volume,rot,
            #                                                                     offset=-((offset.T).flatten()))
            #    transformed_volume = ndimage.interpolation.shift(transformed_volume_no_t,trans)
            # Fix lack of trnaslation
            #    plt.clf()
            #    plt.subplot(2,2,2)
            #    plt.imshow(transformed_volume[:,:,round(width/2)])
            " Calculate homogenous new coordinates"
            for j, element in enumerate(bead):
                bead_pos[j, angle_idx, :] = (
                    rotation_matrix
                    * np.concatenate((np.matrix(bead[j]).T, (np.matrix(1)).T))
                ).flatten()

            first_xy = bead[:, 0:2] / width
            current_xy = bead_pos[:, angle_idx, 0:2] / width
            #    #%%  Essential Matrix Method
            #
            #    ' Find E'
            K = np.matrix("1,0,0;0,1,0;0,0,1")
            #    E,mask = cv2.findEssentialMat(first_xy,current_xy)
            #    R1,R2,t = cv2.decomposeEssentialMat(E)
            #    unit_pose_scaling[angle_idx] = (np.divide(trans,t)).flatten()
            #    points, R_pose, t_pose_unit, mask = cv2.recoverPose(E,first_xy,current_xy)
            #    lhs = np.matrix(current_xy[0]).T-(R_pose*np.matrix(np.concatenate((first_xy[0],np.array([1])))).T)[0:2]
            #
            #    #Cheating step to find which mmatrices are right
            #    square_difference_E_R = np.empty(2)
            #
            #    if sum(sum(((np.array(rot)) - (R1))**2)) < sum(sum(((np.array(rot)) - (R2))**2)):
            #        R_E = np.matrix(R1)
            #    else:
            #        R_E = np.matrix(R2)
            #
            #    t_E = (np.matrix(current_xy[0]).T - (R_E*np.matrix(np.concatenate((first_xy[0],np.array([1])))).T)[0:2])*width
            #    t_E_scale = np.matrix.mean(np.divide(t_E,t_pose_unit[0:2]))
            #    trans_E = np.matrix((t_pose_unit*t_E_scale))
            #
            #    if sum(sum((np.array((trans - trans_E)))**2)) < sum(sum((np.array((trans + trans_E)))**2)):
            #        abs_trans_E = np.matrix(trans_E)
            #    else:
            #        abs_trans_E = np.matrix(-trans_E)
            #    #t_pose = (np.matrix(current_xy[0]).T - (R_pose*np.matrix(np.concatenate((first_xy[0],np.array([1])))).T)[0:2])*width
            ##    E_n = E/(np.sqrt(np.trace(E.T*E))/2)
            ##    U,S,V = np.linalg.svd(E)
            ##    E_norm = np.matrix(U)*np.matrix('1,0,0;0,1,0;0,0,0')*np.matrix(V)
            ##    b_0 = np.sqrt((1-E_norm.T*E_norm)[0,0])
            ##    b_1 = np.sqrt((1-E_norm.T*E_norm)[1,1])
            ##    b_2 = np.sqrt((1-E_norm.T*E_norm)[2,2])
            " Find H"

            H, inliers = cv2.findHomography(first_xy, current_xy)
            a, R, T, translation = cv2.decomposeHomographyMat(H, K)
            T = np.multiply(T, width)
            # Cheating step to find which mmatrices are right
            square_difference_R = np.empty(len(R))
            square_difference_T = np.empty(len(T))
            for count, current_R in enumerate(R):
                square_difference_R[count] = sum(
                    sum(((np.array(rot)) - (current_R)) ** 2)
                )

            for count, current_T in enumerate(T):
                square_difference_T[count] = sum(
                    sum(((np.array(trans)) - (current_T)) ** 2)
                )
            #    for j,element in enumerate(bead):
            #        bead_pos_new[j,angle_idx,:] = ((R.I)*np.concatenate((np.matrix(bead[j]).T,(np.matrix(1)).T))).flatten()
            # %% Cheating step to find which mmatrices are right

            # %% Update rotation matrix
            # rot_square_difference = sum(sum(((np.array(rot)) - np.matrix(R[np.argmin(square_difference_R)]))**2))

            # rot = np.matrix(R[np.argmin(square_difference_R)])

            #    rot  = R_E
            # trans = np.matrix(T[np.argmin(square_difference_T)])
            # trans_square_difference = sum(sum(((np.array(trans)) - (current_R))**2))

            # Square difference of real matrix and found matrix.
            #    trans = abs_trans_E
            #    for j,element in enumerate(bead):
            #        bead_pos_new[j,angle_idx,:] = np.concatenate([np.array(((rot.I)*np.matrix(bead[j]).T)).flatten(),np.array([1])])

            # %% Difference between H and Real
            rot_H = np.array(R[np.argmin(square_difference_R)])
            rot_H[rot_H == 0] = ["nan"]

            trans_H = np.array(T[np.argmin(square_difference_T)])
            trans_H[trans_H == 0] = ["NaN"]

            nan_rot = copy.deepcopy(rot)
            nan_rot[nan_rot == 0] = ["nan"]

            nan_trans = copy.deepcopy(trans)
            nan_trans[nan_trans == 0] = ["nan"]

            rot_diff = nan_rot - rot_H
            rot_sum = nan_rot + rot_H
            pc_rot = np.absolute(np.divide(rot_diff, rot_sum / 2))
            pc_sum_rot[angle_idx, helicity_idx] = np.nansum(pc_rot) / 9

            trans_diff = nan_trans - trans_H
            trans_sum = nan_trans + trans_H
            pc_trans = np.absolute(np.divide(trans_diff, trans_sum / 2))
            pc_sum_trans[angle_idx, helicity_idx] = np.nansum(pc_trans) / 3
            print(f"t_x : {t_x} | alpha: {alpha}")
            print(
                (
                    "angle_idx"
                    + str(angle_idx)
                    + "Rot pc Diff: "
                    + str(pc_sum_rot[angle_idx, helicity_idx])
                    + "  Trans pc Diff: "
                    + str(pc_sum_trans[angle_idx, helicity_idx])
                )
            )

        #    #%% Projection
        #    ' Projection'
        #    projection = np.sum(transformed_volume,axis=0) #Probably axis 2?
        #    sinugram[:,angle_idx] = np.sum(projection,axis=1)
        #    #%%
        #    back_projection = np.tile(projection,(width,1,1)) #check extensively.
        #    #transformed_back_projection = ndimage.interpolation.affine_transform(back_projection,(rotation_matrix.I))
        #    back_projection = ndimage.interpolation.shift(back_projection,-trans)
        #    offset=np.array((centre-centre.dot(rot.I)).dot(np.linalg.inv(rot.I)))
        #    transformed_back_projection = ndimage.interpolation.affine_transform(back_projection,(rot.I),
        #                                                                         order=2,offset=-((offset.T).flatten()))
        #    reconstruction_back_projection = transformed_back_projection + reconstruction_back_projection
        #    drawnow(draw_fig,angle_idx)
        #
        #
        # plt.subplot(2,2,4)
        # plt.imshow(reconstruction_back_projection[:,:,round(width/2)])
        ##
        ##ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
        # plt.show()
        #####
        #    #%%
        #    img = reconstruction_back_projection[:,:,round(width/2)]
        #    f = fftfreq(width).reshape(-1, 1)   # digital frequency
        #    omega = 2 * np.pi * f                                # angular frequency
        #    fourier_filter = 2 * np.abs(f)
        #    projection = fft(img, axis=0) * fourier_filter
        #    filtered = np.real(ifft(projection, axis=0))
    error_types = {"ROT": pc_sum_rot, "TRANS": pc_sum_trans}
    for error_type_key in error_types:

        df = pd.DataFrame(
            error_types[error_type_key], columns=helical_shifts, index=angles
        )
        df["error type"] = error_type_key
        df["condition"] = condition_name
        df.index.names = ["angles"]
        df.drop(df.tail(1).index, inplace=True)
        # df.columns.names = ['Tx']

        df_melt = pd.melt(
            df.reset_index(),
            id_vars=["angles", "error type", "condition"],
            var_name="var",
            value_name="Error",
        )
        df_melt["var"] = pd.to_numeric(df_melt["var"], errors="coerce").round(
            decimals=4
        )
        # df_melt["var"] = df_melt["var"].astype(str).str[0:4]
        # df.round(decimals)

        list_of_dfs[condition_name, error_type_key] = df_melt

# %% Collate dataframes
full_df = pd.concat(list_of_dfs, ignore_index=True)
full_df["angle_str"] = full_df["angles"].map("{:1.2f}".format)
full_df["var_str"] = full_df["var"].map("{:1.2f}".format)
full_df = full_df.set_index(
    ["angles", "error type", "condition", "angle_str", "var_str"]
)


# %% Build graphs
from matplotlib import rc

# rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# rc("font", **{"family": "serif", "serif": ["Palatino"]})
plt.rcParams.update({"font.size": 16})


def plot_violin(condition, error_type, xlabel):
    fig = plt.figure(figsize=(8, 8 / 1.6))
    sns.boxplot(
        x="var_str",
        y="Error",
        data=full_df.xs(
            [condition, error_type], level=["condition", "error type"]
        ).reset_index(),
    )
    plt.xlabel(xlabel)
    plt.ylabel("Absolute error")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(metadata(f"{condition}_{error_type}_err"))
    plt.show()


plot_violin("Tx", "TRANS", "Helical shift /px")
plot_violin("alpha", "TRANS", "$\\alpha $ shift /rad")
plot_violin("Tx", "ROT", "Helical shift /px")
plot_violin("alpha", "ROT", "$\\alpha $ shift /rad")

# %%
