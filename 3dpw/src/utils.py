import os, sys, re, pickle, json
import numpy as np
import cv2
import pandas as pd

def get_seq(seq_dir, seq_name):
    seq_file = seq_dir + "/" + seq_name + ".pkl"

    seq = pickle.load(open(seq_file, "rb"), encoding='latin1')

    return seq

def get_3dkeypoints(seq, frame_id, model_id):
    """
    SMPL joints
    0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
    :param seq:
    :param frame_id:
    :param model_id:
    :return:
    """

    _3d_keypoints = seq["jointPositions"][model_id][frame_id]

    _3d_keypoints = _3d_keypoints.reshape(-1, 3)

    return _3d_keypoints

def get_cam_params(seq, frame_id):

    intrinsic = seq["cam_intrinsics"]

    extrinsic = seq["cam_poses"][frame_id]

    R = extrinsic[:3,:3]
    t = extrinsic[:-1, -1]

    t = np.expand_dims(t, axis=1)

    return intrinsic, R, t, extrinsic

def estimate_from_3d(seq, frame_id, model_id):

    keypoints3d = get_3dkeypoints(seq, frame_id, model_id)

    intrinsic, R, t, extrinsic = get_cam_params(seq, frame_id)

    estimated_keypoints_2d, _ = cv2.projectPoints(keypoints3d, R, t, intrinsic, None)

    estimated_keypoints_2d = np.squeeze(estimated_keypoints_2d, axis=1)

    return estimated_keypoints_2d


def approximate_bb(keypoints):
    x_offset = 10
    y_offset = 10

    xs = keypoints.T[0]
    ys = keypoints.T[1]

    x_min = int(xs.min()) - x_offset
    x_max = int(xs.max()) + x_offset

    y_min = int(ys.min()) - y_offset
    y_max = int(ys.max()) + y_offset

    top_left = [x_min, y_min]

    bottom_right = [x_max, y_max]

    return top_left, bottom_right

def smpl2coco(smpl_pose):
    """
    smpl_format = ["pelvis", "left_hip", "right_hip", "lower_spine", "left_knee", "right_knee", # 0-5
        "middle_spine", "left_ankle", "right_ankle", "upper_spine", "left_foot", "right_foot", # 6-11
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", # 12-17
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"] # 18-23

    coco_format =  ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder',
        'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle',
        'left_hip', 'left_knee', 'left_ankle', 'head', 'neck', 'right_ear',
        'left_ear', 'nose', 'right_eye', 'left_eye']
    """
    offset = 0
    num_models = len(smpl_pose)

    coco_poses = np.zeros((num_models, 19, 2))
    
    #(smpl, coco)
    common_joints = [(1, 9), (2, 6), (4, 10), (5, 7), (7, 11), (8, 8), (12, 13), (15, 12), (16, 3), (17, 0), (18, 4), (19, 1), (20, 5), (21, 2)]

    for model_id in range(num_models):

        for (smpl_joint, coco_joint) in common_joints:
            coco_poses[model_id][coco_joint] = smpl_pose[model_id][smpl_joint]

        coco_poses[model_id][14] = coco_poses[model_id][12] + offset # right_ear = head
        coco_poses[model_id][15] = coco_poses[model_id][12] + offset # left_ear = head
        coco_poses[model_id][16] = coco_poses[model_id][12] + offset # nose = head
        coco_poses[model_id][17] = coco_poses[model_id][12] + offset # right_eye = head  
        coco_poses[model_id][18] = coco_poses[model_id][12] + offset # left_eye = head

    return coco_poses


def dump_sorted(filename_list, index_list, occ_status, subset_name, scene_name="courtyard_basketball_00", folder_name = "./3dpw/selected_frames"):

    selected = zip(filename_list, index_list, occ_status)
    
    selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True) # sort by occlusion value in descending order
    os.makedirs(folder_name + "/" + scene_name, exist_ok=True)

    with open(folder_name + "/" + scene_name + "/" + subset_name+".txt", "w+") as dump_file :
    
        for result in selected_sorted:
            dump_file.write(
                "{} {} #{}\n".format(result[0], result[1], result[2])
            )