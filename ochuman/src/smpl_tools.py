import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation
from utils import get_json

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


def read_smpl(prefix, split="train"):
    train_path = prefix + "smpl_labels/OCHuman_train_ver10.json"
    test_path = prefix + "smpl_labels/OCHuman_test_ver10.json"

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