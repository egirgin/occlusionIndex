import numpy as np
import cv2
from enum import Enum

def occlusion_index(masks, keypoints):
    num_models = len(keypoints)

    occluded_keypoints_list = np.zeros((num_models, 19), dtype=bool) # 1 means occluded, 0 means visible 

    missing_keypoints_list = np.zeros((num_models, 19), dtype=int) # 1 means missing, 0 means present

    for model_id in range(num_models): # this brings keypoints

        for other_model_id in range(num_models): # this brings masks
            if model_id == other_model_id:
                continue

            for kp_id, kp in enumerate(keypoints[model_id]):
                for mask in masks[other_model_id]:
                    mask = np.array(mask, np.int32)
                    mask = mask.reshape((-1,2))
                    mask = mask.reshape((-1, 1, 2))

                    if (int(kp[0]), int(kp[1])) == (0, 0):
                        missing_keypoints_list[model_id, kp_id] = True
                    
                    elif cv2.pointPolygonTest(mask, (int(kp[0]), int(kp[1])), False) == 1:

                        occluded_keypoints_list[model_id, kp_id] = True
            

    return occluded_keypoints_list, missing_keypoints_list


joint_list =  ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder',
                'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle', 
                'left_hip', 'left_knee', 'left_ankle', 'head', 'neck', 'right_ear', 
                'left_ear', 'nose', 'right_eye', 'left_eye'] 

