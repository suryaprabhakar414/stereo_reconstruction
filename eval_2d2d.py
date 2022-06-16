import numpy as np
import scipy
import sys
import os
import math

T_02 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])
T_03 = np.array([-4.731050e-01, 5.551470e-03, -5.250882e-03])
R_02 = np.array([9.999758e-01, -5.267463e-03, -4.552439e-03, 5.251945e-03, 9.999804e-01, -3.413835e-03, 4.570332e-03, 3.389843e-03, 9.999838e-01]).reshape(3, 3)
R_03 = np.array([9.995599e-01, 1.699522e-02, -2.431313e-02, -1.704422e-02, 9.998531e-01, -1.809756e-03, 2.427880e-02, 2.223358e-03, 9.997028e-01]).reshape(3, 3)

T_23 = -T_02 + T_03
R_23 = R_02.T.dot(R_03)
T_23_norm = T_23 / np.linalg.norm(T_23)

def rotation_error(pose_in):
    pose = pose_in @ R_23.T
    a = pose[0][0]
    b = pose[1][1]
    c = pose[2][2]
    d = 0.5*(a+b+c-1.0)
    return math.acos(max(min(d,1.0),-1.0))

def translation_error(pose):
    dx = pose[0][3] - T_23_norm[0]
    dy = pose[1][3] - T_23_norm[1]
    dz = pose[2][3] - T_23_norm[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

if len(sys.argv) < 3:
    var_list = ['orb.txt', 'sift.txt', 'surf.txt']
else:
    var_list = sys.argv[1:]

for filename in var_list:
    path = os.path.join('/Users/yang/ws/stereo-reconstruction/result/SGBM', filename)
    with open(path, 'r') as f:
        print(filename[:-4].upper())
        t_rms = np.zeros((3, 1))
        r_rms = np.zeros((3, 1))
        r_errs = []
        t_errs = []
        data = np.loadtxt(f, delimiter=',')
        for _transform in data:
            transform = _transform.reshape((4, 4))
            r_err = rotation_error(transform[:3, :3])
            t_err = translation_error(transform)
            r_errs.append(r_err)
            t_errs.append(t_err)

        n_data = data.shape[0]
        print("Translation Error on {} poses: {}".format(n_data, np.linalg.norm(np.array(t_errs)) / np.sqrt(n_data)))
        print("Rotation Error on {} poses: {}".format(n_data, np.linalg.norm(np.array(r_errs)) / np.sqrt(n_data)))
