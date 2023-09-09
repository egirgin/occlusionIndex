import numpy as np
import cv2
from enum import Enum

def occlusion_index(masks, keypoints):
    num_models = len(keypoints)

    occlusion_indices = []
    missing_indices = []

    occluded_keypoints_list = np.zeros((num_models, 19), dtype=int) # 1 means occluded, 0 means visible 

    missing_keypoints_list = np.zeros((num_models, 19), dtype=int) # 1 means missing, 0 means present


    for model_id in range(num_models): # this brings keypoints
        n_occluded_keypoints = 0
        n_missing_keypoints = 0

        for other_model_id in range(num_models): # this brings masks
            if model_id == other_model_id:
                continue

            for kp_id, kp in enumerate(keypoints[model_id]):
                for mask in masks[other_model_id]:
                    mask = np.array(mask, np.int32)
                    mask = mask.reshape((-1,2))
                    mask = mask.reshape((-1, 1, 2))

                    if (int(kp[0]), int(kp[1])) == (0, 0):
                        n_missing_keypoints += 1

                        missing_keypoints_list[model_id, kp_id] = 1
                    
                    elif cv2.pointPolygonTest(mask, (int(kp[0]), int(kp[1])), False) == 1:
                        n_occluded_keypoints += 1

                        occluded_keypoints_list[model_id, kp_id] = 1
            
        occlusion_indices.append(n_occluded_keypoints / 19) # coco format has 19 keypoints
        missing_indices.append(n_missing_keypoints)

    return occlusion_indices, occluded_keypoints_list, missing_indices, missing_keypoints_list


class COCO_JOINT(Enum):
    RIGHT_SHOULDER = 0
    RIGHT_ELBOW = 1
    RIGHT_WRIST = 2
    
    LEFT_SHOULDER = 3
    LEFT_ELBOW = 4
    LEFT_WRIST = 5

    RIGHT_HIP = 6
    RIGHT_KNEE = 7
    RIGHT_ANKLE = 8

    LEFT_HIP = 9
    LEFT_KNEE = 10
    LEFT_ANKLE = 11
    
    HEAD = 12
    NECK = 13

    RIGHT_EAR = 14
    LEFT_EAR = 15

    NOSE = 16

    RIGHT_EYE = 17
    LEFT_EYE = 18

class COCO_SUBSET(Enum):

    right_upper_subset = [COCO_JOINT.RIGHT_SHOULDER, COCO_JOINT.RIGHT_ELBOW, COCO_JOINT.RIGHT_WRIST]
    left_upper_subset = [COCO_JOINT.LEFT_SHOULDER, COCO_JOINT.LEFT_ELBOW, COCO_JOINT.LEFT_WRIST]

    right_lower_subset = [COCO_JOINT.RIGHT_HIP, COCO_JOINT.RIGHT_KNEE, COCO_JOINT.RIGHT_ANKLE]
    left_lower_subset = [COCO_JOINT.LEFT_HIP, COCO_JOINT.LEFT_KNEE, COCO_JOINT.LEFT_ANKLE]

    head_subset = [COCO_JOINT.HEAD, COCO_JOINT.NECK, COCO_JOINT.NOSE]
    torso_subset = [COCO_JOINT.RIGHT_SHOULDER, COCO_JOINT.RIGHT_HIP, COCO_JOINT.LEFT_SHOULDER, COCO_JOINT.LEFT_HIP]

coco_subset = {
    "right_upper_subset" : [COCO_JOINT.RIGHT_SHOULDER, COCO_JOINT.RIGHT_ELBOW, COCO_JOINT.RIGHT_WRIST],
    "left_upper_subset" : [COCO_JOINT.LEFT_SHOULDER, COCO_JOINT.LEFT_ELBOW, COCO_JOINT.LEFT_WRIST],

    "right_lower_subset" : [COCO_JOINT.RIGHT_HIP, COCO_JOINT.RIGHT_KNEE, COCO_JOINT.RIGHT_ANKLE],
    "left_lower_subset" : [COCO_JOINT.LEFT_HIP, COCO_JOINT.LEFT_KNEE, COCO_JOINT.LEFT_ANKLE],

    "head_subset" : [COCO_JOINT.HEAD, COCO_JOINT.NECK, COCO_JOINT.NOSE],
    "torso_subset" : [COCO_JOINT.RIGHT_SHOULDER, COCO_JOINT.RIGHT_HIP, COCO_JOINT.LEFT_SHOULDER, COCO_JOINT.LEFT_HIP],
    
    "left_body_subset" : COCO_SUBSET.left_upper_subset.value + COCO_SUBSET.left_lower_subset.value,
    "right_body_subset" : COCO_SUBSET.right_upper_subset.value + COCO_SUBSET.right_lower_subset.value,
    "upper_body_subset" : COCO_SUBSET.right_upper_subset.value + COCO_SUBSET.left_upper_subset.value,
    "lower_body_subset" : COCO_SUBSET.left_upper_subset.value + COCO_SUBSET.left_lower_subset.value,

    "empty_subset" : []
}


def form_criterion(subset):
    criterion = np.zeros(19, dtype=bool)

    for i in range(19):
        for joint in subset:
            criterion[joint.value] = True

    return criterion

joint_list =  ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder',
                'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle', 
                'left_hip', 'left_knee', 'left_ankle', 'head', 'neck', 'right_ear', 
                'left_ear', 'nose', 'right_eye', 'left_eye'] 

def filter_by_criterion(criterion, occlusion_status):
    for model_occlusion in occlusion_status:
        
        if all(model_occlusion[criterion]):
            #print(model_occlusion[criterion])
            return True
    
    return False