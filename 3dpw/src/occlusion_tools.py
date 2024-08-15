import numpy as np
from visualize import *
from utils import *
from enum import Enum

def occlusion_index(mask, keypoints, colormap=color_list):
    num_models = len(keypoints)    
    
    occluded_keypoints_list = np.zeros((num_models, 19), dtype=int) # 1 means occluded, 0 means visible 
    occlusion_indices = []
     
    for model_id in range(num_models): # this brings keypoints

        model_color = colormap[model_id]

        n_occluded_keypoints, occlusion_status = occlusion_helper(mask=mask, keypoints=keypoints[model_id], model_color=model_color)

        occluded_keypoints_list[model_id] = occlusion_status

        occlusion_indices.append(n_occluded_keypoints / 19) # coco pose format has 19 kps

    return occlusion_indices, occluded_keypoints_list

def occlusion_helper(mask, keypoints, model_color):

    # keypoints: occluded
    # person color code: occluded
    height, width = mask.shape[:2]
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    if np.all(xs <= 0) or np.all(xs >= height) or np.all(ys <= 0) or np.all(ys >= width):
        # if one of the models is completely outside of the canvas, count it as all visible 
        return 0, np.zeros(19, dtype=int)

    occluded_keypoints = np.zeros(19, dtype=int) # 1 means occluded, 0 means visible 

    for kp_id, keypoint in enumerate(keypoints):
        pixel_coords = [int(np.floor(keypoint[1])), int(np.floor(keypoint[0]))]
        
        # if a keypoint is out of the canvas, count it as occluded
        if pixel_coords[0] <= 0 or pixel_coords[0] >= height or pixel_coords[1] <= 0 or pixel_coords[1] >= width:
            occluded_keypoints[kp_id] = 1
            continue

        mask_value = mask[pixel_coords[0], pixel_coords[1]]

        distance2colors = color_list - mask_value
        distance2colors = np.sum(np.abs(distance2colors), axis=1)

        if not np.array_equal(color_list[np.argmin(distance2colors)], model_color):
            occluded_keypoints[kp_id] = 1

    return np.sum(occluded_keypoints), occluded_keypoints


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
        
        if np.all(model_occlusion[criterion]):
            #print(model_occlusion[criterion])
            return True
    
    return False


def crowd_metric(bbox, keypoints):

    intersection_keypoints = []

    count = 0
    for keypoint in keypoints:
        if bbox[0][0] < keypoint[0] < bbox[1][0] and bbox[0][1] < keypoint[1] < bbox[1][1]:
            count += 1
            intersection_keypoints.append([keypoint[0], keypoint[1]])

    return count, intersection_keypoints
