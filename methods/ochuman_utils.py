import numpy as np
import json


def coco2smpl(coco_pose):
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

    num_models = len(coco_pose)

    smpl_poses = np.zeros((num_models, 24, 2), dtype=bool)

    common_joints = [(1, 9), (2, 6), (4, 10), (5, 7), (7, 11), (8, 8), (12, 13), (15, 12), (16, 3), (17, 0), (18, 4), (19, 1), (20, 5), (21, 2)]

    for model_id in range(num_models):

        for (smpl_joint, coco_joint) in common_joints:
            smpl_poses[model_id][smpl_joint] = coco_pose[model_id][coco_joint]

        smpl_poses[model_id][0] = np.mean([smpl_poses[model_id][1] , smpl_poses[model_id][2]], axis=0) # pelvis = left_hip and right_hip
        smpl_poses[model_id][3] = smpl_poses[model_id][0] # lower_spine = pelvis
        smpl_poses[model_id][9] = np.mean([smpl_poses[model_id][16] , smpl_poses[model_id][17]], axis=0) # upper_spine = left_shoulder and right_shoulder
        smpl_poses[model_id][6] = np.mean([smpl_poses[model_id][3] , smpl_poses[model_id][9]], axis=0) # middle_spine = lower_spine and upper_spine
        smpl_poses[model_id][10] = smpl_poses[model_id][7] # left_foot = left_ankle
        smpl_poses[model_id][11] = smpl_poses[model_id][8] # right_foot = right_ankle
        smpl_poses[model_id][13] =np.mean([ smpl_poses[model_id][16] , smpl_poses[model_id][12]], axis=0) # left_collar = left_shoulder and neck
        smpl_poses[model_id][14] =np.mean([ smpl_poses[model_id][17] , smpl_poses[model_id][12]], axis=0) # right_collar = right_shoulder and neck
        smpl_poses[model_id][22] = smpl_poses[model_id][20] # left_hand = left_wrist
        smpl_poses[model_id][23] = smpl_poses[model_id][21] # right_hand = right_wrist

    return smpl_poses


def smpl2coco(smpl_poses):
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
    num_models = len(smpl_poses)

    coco_poses = np.zeros((num_models, 14, 2), dtype=float)

    # smpl, coco
    common_joints = [(1, 9), (2, 6), (4, 10), (5, 7), (7, 11), (8, 8), (12, 13), (15, 12), (16, 3), (17, 0), (18, 4), (19, 1), (20, 5), (21, 2)]

    for model_id in range(num_models):

        for (smpl_joint, coco_joint) in common_joints:
            coco_poses[model_id][coco_joint] = smpl_poses[model_id][smpl_joint]

    return coco_poses



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


def get_smpl(imgname):
    try:
        with open("./ochuman/gtSMPL/"+imgname[:-4]+".txt", "r") as smplGT:
            smpls = smplGT.read().splitlines()
    except:
        return None, None

    return len(smpls), smpls