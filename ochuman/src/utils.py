import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation
from operator import itemgetter


def get_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    return data

def get_annos(path):
    train_json = path
    #test_json = "../ochuman_coco_format_test_range_0.00_1.00.json"
    #val_json = "../ochuman_coco_format_val_range_0.00_1.00.json"

    data = get_json(train_json)

    return data



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

def estimate_bbox(keypoints):
    x_offset = 20
    y_offset = 20

    nonzero_keypoints = keypoints[np.nonzero(keypoints)[0]]

    left_top = [int(nonzero_keypoints[:, 0].min() - x_offset), int(nonzero_keypoints[:, 1].min() - y_offset)]

    right_bottom = [int(keypoints[:, 0].max() + x_offset), int(keypoints[:, 1].max() + y_offset)]

    return np.array([left_top, right_bottom])

def crowd_index(keypoints):
    num_models = len(keypoints)

    crowd_indices = [] # holds values per person

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
                    crowd_keypoints.append(kp_id)

        crowd_indices.append(len(crowd_keypoints) / 19) # coco format has 19 keypoints

    return crowd_indices

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


def dump_sorted(filename_list, index_list, occ_status, subset_name, folder_name = "../selected_frames"):

    selected = zip(filename_list, index_list, occ_status)
    
    selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True) # sort by occlusion value in descending order
    os.makedirs(folder_name, exist_ok=True)

    with open(folder_name + "/" + subset_name+".txt", "w+") as dump_file :
    
        for result in selected_sorted:
            dump_file.write(
                "{} {} #{}\n".format(result[0], result[1], result[2])
            )

