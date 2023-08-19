import os
import json
import pickle
import numpy as np
import pandas as pd # pip install "pandas<2.0.0"


cam_path = "/home/emre/Documents/master/thesis/agora/data/cam"


def get_cam(cam_path, pkl_id=None):
    """
    :param cam_path:
    :param pkl_id:
    :return: pandas DF
        columns:
        Index(['X', 'Y', 'Z', 'Yaw', 'imgPath', 'camYaw', 'camZ', 'camY', 'camX',
           'gender', 'gt_path_smpl', 'gt_path_smplx', 'kid', 'occlusion',
           'isValid', 'age', 'ethnicity'],
          dtype='object')

        check readme for details

        ['X', 'Y', 'Z', 'Yaw', 'imgPath', 'camYaw', 'camZ', 'camY', 'camX',
       'gender', 'gt_path_smpl', 'gt_path_smplx', 'kid', 'occlusion',
       'isValid', 'age', 'ethnicity', 'gt_joints_2d', 'gt_joints_3d',
       'gt_verts']
    """
    
    foldername = cam_path+"/validation_annos" 

    os.makedirs(foldername, exist_ok=True)


    appended_data = []
    for pkl_id in range(10):

        path = cam_path + "/validation_SMPL/SMPL/validation_" + str(pkl_id) + "_withjv" + ".pkl"

        with open(path, "rb") as cam_file:
            df = pickle.load(cam_file)

            appended_data.append(df)

    appended_data = pd.concat(appended_data)

    for idx, row in appended_data.iterrows():
        record = {
            "smpl_path" : row["gt_path_smpl"],
            "smplx_path" : row["gt_path_smplx"],
            "keypoints2d" : [model.tolist() for model in row["gt_joints_2d"] ],  # Nx45x2 # convert to COCO format explicitly
            "keypoints3d" : [model.tolist() for model in row["gt_joints_3d"] ], # Nx45x3 
            "location" : list(zip(row["X"], row["Y"], row["Z"], row["Yaw"])), # Nx4 (X,Y,Z,Yaw) yaw in degrees
            "cam_extrinsics" : [row["camX"], row["camY"], row["camZ"], row["camYaw"]] # 1x4 (X,Y,Z,Yaw)
        }

        annotation_filename = row["imgPath"][:-4] + ".json"

        with open(foldername + "/" + annotation_filename, "w+") as annotation:
            json.dump(record, annotation)

    return appended_data


get_cam(cam_path=cam_path)
