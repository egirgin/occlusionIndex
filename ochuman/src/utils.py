import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation


def get_annos(split="train"):
    train_json = "../ochuman.json"
    test_json = "../ochuman_coco_format_test_range_0.00_1.00.json"
    val_json = "../ochuman_coco_format_val_range_0.00_1.00.json"


    data = get_json(train_json)

    return data



def get_frame_annos(data, frame_id):
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
            print("Frame {} has missing data".format(frame_id))
            return None, None, None

        # a model may have more than one outer mask since occlusion
        masks.append(
            data["images"][frame_id]["annotations"][model_id]["segms"]["outer"]
        )

        if not data["images"][frame_id]["annotations"][model_id]["segms"]["inner"] == []:
            print("Inner not empty!!! Frame id: {}, model id: {}".format(frame_id, model_id))
            #masks.append(
            #    data["images"][frame_id]["annotations"][model_id]["segms"]["inner"]
            #)

        # keypoint format -> (x,y,c) c=0,1,2 vis, invis, missing
        keypoint = data["images"][frame_id]["annotations"][model_id]["keypoints"]
        if len(keypoint) < 19*3:
            print("keypoints missing")

        keypoints.append(np.array_split(keypoint, 19))

    return img_filename, masks, np.array(keypoints)
def eft2smplPose(eft_pose):
    eft_pose_reshape = np.reshape(eft_pose, (24, 9))

    # Initialize the SMPL pose parameters matrix
    smpl_pose = np.zeros((24, 3))

    # Compute axis-angle representation for each joint rotation
    for i in range(24):
        rotmat = eft_pose_reshape[i].reshape(3, 3)
        r = Rotation.from_matrix(rotmat)
        euler = r.as_euler("xyz")
        smpl_pose[i] = euler


        """
        rotvec = r.as_rotvec()
        axis = rotvec[1:]
        if np.linalg.norm(axis) == 0:
            angle = 0
        else:
            angle = rotvec[0]
            axis = axis / np.linalg.norm(axis)
        smpl_pose[i] = np.concatenate(([angle], axis))
        """

    return smpl_pose


def get_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    return data

def read_smpl(split="train"):
    train_path = "../smpl_labels/OCHuman_train_ver10.json"
    test_path = "../smpl_labels/OCHuman_test_ver10.json"

    if split == "train":
        smpl = get_json(train_path)
    elif split == "test":
        smpl = get_json(test_path)
    else:
        return None
    return smpl

def process_smpl(smpl_json):
    os.makedirs("gtSMPL", exist_ok=True)

    for data in smpl_json["data"]:

        smpl_dict = {
            "imageName": data["imageName"],
            "parm_pose": eft2smplPose(data["parm_pose"]).tolist(),
            "parm_shape": data["parm_shape"],
            "parm_cam": data["parm_cam"],
            "bbox_center": data["bbox_center"]
        }

        with open("gtSMPL/"+data["imageName"][:-3]+"txt", "a+") as gt_json:
            str_json = json.dumps(smpl_dict)
            gt_json.write(str_json+"\n")


def get_smpl(imgname):
    try:
        with open("gtSMPL/"+imgname[:-4]+".txt", "r") as smplGT:
            smpls = smplGT.read().splitlines()
    except:
        return None, None

    return len(smpls), smpls
