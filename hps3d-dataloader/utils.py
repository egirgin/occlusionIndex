import numpy as np
import cv2
import os, json
from enum import Enum


def read_img(img_path, scale=1):
    img = cv2.imread(img_path)

    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    return img


def get_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    return data

def estimate_bbox(keypoints):
    x_offset = 20
    y_offset = 20

    nonzero_keypoints = keypoints[np.nonzero(keypoints)[0]]

    left_top = [int(nonzero_keypoints[:, 0].min() - x_offset), int(nonzero_keypoints[:, 1].min() - y_offset)]

    right_bottom = [int(keypoints[:, 0].max() + x_offset), int(keypoints[:, 1].max() + y_offset)]

    return np.array([left_top, right_bottom])


def occlusion_index(masks, keypoints):
    num_models = len(keypoints)

    occlusion_indices = []
    missing_indices = []

    occluded_keypoints_list = np.zeros((num_models, 19), dtype=int) # zero means visible, one means occluded

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

def crowd_index(keypoints):
    num_models = len(keypoints)

    crowd_indices = [] # holds values per person

    crowd_keypoints_list = np.zeros((num_models, 19), dtype=int) # zero means visible, one means occluded

    for model_id in range(num_models):
        bbox = estimate_bbox(keypoints[model_id])

        crowd_keypoints = []

        for other_model_id in range(num_models):
            if model_id == other_model_id:
                continue

            for kp_id, kp in enumerate(keypoints[other_model_id]):
                left_top = bbox[0]
                right_bottom = bbox[1]

                if right_bottom[0] > kp[0] > left_top[0] and right_bottom[1] > kp[1] > left_top[1]:
                    crowd_keypoints_list[model_id][kp_id] = 1

        crowd_indices.append(len(crowd_keypoints) / 19) # coco format has 19 keypoints

    return crowd_indices, crowd_keypoints_list


def ochuman_annotation(data, frame_id, verbose=0):
    ####!!!!!!!!!!!!1 BEAWARE THAT FRAME ID != IMG FILENAME

    # print(data.keys())
    # print(len(data["images"]))
    # print(data["images"][0].keys())
    # print(data["images"][0]["annotations"][0].keys())
    # print(data["images"][0]["annotations"][0]["segms"])
    # print(data["images"][0]["annotations"][0]["bbox"])
    # print(data["images"][0]["annotations"][0]["segms"].keys())
    # print(data["images"][0]["annotations"][0]["segms"]["outer"])
    # print(len(data["keypoint_names"]))
    # print(data["keypoint_names"])
    # print(data["keypoint_visible"]) # {'vis': 1, 'others_occluded': 3, 'self_occluded': 2, 'missing': 0}

    num_models = len(data["images"][frame_id]["annotations"])

    img_filename = data["images"][frame_id]["file_name"]

    masks = []
    keypoints = []

    for model_id in range(num_models):
        if data["images"][frame_id]["annotations"][model_id]["segms"] == None or \
                data["images"][frame_id]["annotations"][model_id]["keypoints"] == None:
            if verbose == 1:
                print("Frame {} has missing data".format(frame_id))
            return None, None, None

        # a model may have more than one outer mask since occlusion
        masks.append(
            data["images"][frame_id]["annotations"][model_id]["segms"]["outer"]
        )

        if not data["images"][frame_id]["annotations"][model_id]["segms"]["inner"] == []:
            if verbose == 1:
                print("Inner not empty!!! Frame id: {}, model id: {}".format(frame_id, model_id))

            # masks.append(
            #    data["images"][frame_id]["annotations"][model_id]["segms"]["inner"]
            # )
            pass

        # keypoint format -> (x,y,c) c=0,1,2 vis, invis, missing
        keypoint = data["images"][frame_id]["annotations"][model_id]["keypoints"]
        if len(keypoint) < 19 * 3:
            print("keypoints missing")

        keypoints.append(np.array_split(keypoint, 19))

    return img_filename, masks, np.array(keypoints)


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


def form_criterion(subset):
    criterion = np.zeros(19)

    for i in range(19):
        for joint in subset.value:
            criterion[joint.value] = 1

    return criterion

joint_list =  ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder',
                'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle', 
                'left_hip', 'left_knee', 'left_ankle', 'head', 'neck', 'right_ear', 
                'left_ear', 'nose', 'right_eye', 'left_eye'] 