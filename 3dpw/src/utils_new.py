import trimesh
import json, sys
import numpy as np
from enum import Enum

color_list = [
    [255, 0, 0], # red
    [0, 0, 255], # blue
    [179, 179, 179], # pink
    [230, 230, 230], # neutral
    [179, 192, 128], # capsule
    [128, 179, 192] # yellow
]

def joint_names(part_segm_filepath):
    part_segm = json.load(open(part_segm_filepath))

    smpl_joints = []
    for joint_name in part_segm.keys():
        smpl_joints.append(joint_name)
    return smpl_joints

def occlusion_index(mask, keypoints, colormap=color_list):
    num_models, num_joints = keypoints.shape[:2]

    occluded_keypoints_list = np.zeros((num_models, num_joints), dtype=bool)

    truncated_keypoints_list = np.zeros((num_models, num_joints), dtype=bool)
     
    for model_id in range(num_models): # this brings keypoints

        model_color = colormap[model_id]

        occlusion_mask, truncation_mask = occlusion_helper(mask=mask, keypoints=keypoints[model_id], model_color=model_color)

        occluded_keypoints_list[model_id] = occlusion_mask

        truncated_keypoints_list[model_id] = truncation_mask

    return occluded_keypoints_list, truncated_keypoints_list

def occlusion_helper(mask, keypoints, model_color):

    # keypoints: occluded
    # person color code: occluded
    image_height, image_width = mask.shape[:2]
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
    num_joints = len(xs)

    joint_occlusion_mask = np.zeros(num_joints, dtype=bool)
    truncation = np.zeros(num_joints, dtype=bool) 

    for kp_id, keypoint in enumerate(keypoints):
        y = int(np.floor(keypoint[1]))
        x = int(np.floor(keypoint[0]))
        if not image_height > y >= 0 or not image_width > x >= 0:
            truncation[kp_id] = True
            continue

        pixel_color = mask[y, x]

        distance2colors = color_list - pixel_color
        distance2colors = np.sum(np.abs(distance2colors), axis=1)

        if not np.array_equal(color_list[np.argmin(distance2colors)], model_color):
            joint_occlusion_mask[kp_id] = True

    return joint_occlusion_mask, truncation

def crowd_metric(bbox, keypoints, current_model_id):
    
    num_models, num_joints = keypoints.shape[:2]

    crowd_index = 0

    for model_id in range(num_models):
        if model_id == current_model_id:
            continue

        for joint_id in range(num_joints):
            keypoint = keypoints[model_id][joint_id]

            if bbox[0][0] < keypoint[0] < bbox[1][0] and bbox[0][1] < keypoint[1] < bbox[1][1]:
                crowd_index += 1
        
    return crowd_index / num_joints


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

    "all_subset" : []
}