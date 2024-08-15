import os, sys
import numpy as np
import pickle, json
import cv2
from smpl_np_romp import SMPLModel

dataset_folder = "/home/tuba/Documents/emre/thesis/dataset/3dpw"
out_folder = "./3dpw/smpl_objects"

class cam:
    pass

def get_seq(seq_name, subfolder="train"):
    seq_dir = dataset_folder + "/sequenceFiles/sequenceFiles/{}".format(subfolder)

    seq_file = seq_dir + "/" + seq_name + ".pkl"

    seq = pickle.load(open(seq_file, "rb"), encoding='latin1')

    return seq


def get_img(seq_name, frame_id, scale):
    img_dir = dataset_folder + "/imageFiles"

    img_file = img_dir + "/" + seq_name + "/image_" + str(frame_id).zfill(5) + ".jpg"

    img = cv2.imread(img_file)

    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))


    return img

def get_2d_keypoints(seq, frame_id, model_id):
    """
            "keypoints": [
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"
        ],
    :param seq:
    :param frame_id:
    :param model_id:
    :return:
    """

    poses2d = seq["poses2d"][model_id][frame_id]

    xs = poses2d[0][poses2d[0] > 0]
    ys = poses2d[1][poses2d[1] > 0]
    return np.array([xs, ys])

def approximate_bb(pose2d):
    x_offset = 14
    y_offset = 100

    xs = pose2d[0]
    ys = pose2d[1]

    x_min = int(xs.min()) - x_offset
    x_max = int(xs.max()) + x_offset

    y_min = int(ys.min()) - y_offset
    y_max = int(ys.max()) + y_offset

    top_left = [x_min, y_min]

    bottom_right = [x_max, y_max]

    return top_left, bottom_right


def crowd_metric(bbox, keypoints):

    collison_keypoints_x = []
    collison_keypoints_y = []

    count = 0
    for keypoint in keypoints.T:
        if bbox[0][0] < keypoint[0] < bbox[1][0] and bbox[0][1] < keypoint[1] < bbox[1][1]:
            count += 1
            collison_keypoints_x.append(keypoint[0])
            collison_keypoints_y.append(keypoint[1])

    return count, [collison_keypoints_x, collison_keypoints_y]

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

    intrinsic, R, t = get_cam_params(seq, frame_id)

    estimated_keypoints_2d, _ = cv2.projectPoints(keypoints3d, R, t, intrinsic, None)

    estimated_keypoints_2d = np.squeeze(estimated_keypoints_2d, axis=1).T

    return estimated_keypoints_2d

def get_smpl_betas(seq, model_id):
    betas = seq["betas"][model_id]

    betas_clothed = seq["betas_clothed"][model_id]

    gender = seq["genders"][model_id]

    return betas, betas_clothed, gender

def get_smpl_thetas(seq, model_id, frame_id):

    thetas = seq["poses"][model_id][frame_id]

    trans = seq["trans"][model_id][frame_id]

    return thetas, trans

if __name__ == "__main__":

    seq_list = os.listdir(dataset_folder + "/imageFiles")
    seq_list.sort()

    only_multiperson = True

    seq_list = ["outdoors_freestyle_00"] # cheating to process only one seq

    for seq_name in seq_list:
        print("-------------------------------------------------------")
        print("Processing Sequence {}".format(seq_name))

        img_folder = os.listdir(dataset_folder + "/imageFiles/"+seq_name)
        
        if seq_name+".pkl" in os.listdir(dataset_folder + "/sequenceFiles/sequenceFiles/train"):
            seq = get_seq(seq_name, subfolder="train")
        elif seq_name+".pkl" in os.listdir(dataset_folder + "/sequenceFiles/sequenceFiles/validation"):
            seq = get_seq(seq_name, subfolder="validation")
        elif seq_name+".pkl" in os.listdir(dataset_folder + "/sequenceFiles/sequenceFiles/test"):
            seq = get_seq(seq_name, subfolder="test")
        else:
            print("Sequence {} could not be found!".format(seq_name))
            sys.exit()

        #if only_multiperson and len(seq["poses"]) < 2:
        #    print("Skipping {} due to lack of multi person occlusion.".format(seq_name))
        #    continue 

        smpl_m = SMPLModel('/home/tuba/Documents/emre/thesis/models/converted/SMPL_MALE.pkl')
        smpl_f = SMPLModel('/home/tuba/Documents/emre/thesis/models/converted/SMPL_FEMALE.pkl')

        for frame_id in range(len(img_folder)): #range(0, 1):
            #print(frame_id)

            folder_name = out_folder + "/" + seq_name +  "/image_{}".format(str(frame_id).zfill(5))

            os.makedirs(folder_name, exist_ok=True)

            intrinsic, R, t, extrinsic = get_cam_params(seq, frame_id)

            camera_matrices = {
                "intrinsics": intrinsic.tolist(),
                "extrinsics": extrinsic.tolist()
            }

            with open(folder_name + "/cam.json", "w+") as cam_file:
                json.dump(camera_matrices, cam_file)

            for model_id in range(len(seq["poses"])):

                betas, clothed_betas, gender = get_smpl_betas(seq, model_id)
                if gender == "f":
                    smpl = smpl_f
                elif gender == "m":
                    smpl = smpl_m
                else:
                    print("Gender not defined...")
                    sys.exit()

                beta = betas[:10].reshape(smpl.beta_shape)  # (np.random.rand(*smpl.beta_shape) - 0.5) * 0.06

                thetas, trans = get_smpl_thetas(seq, model_id, frame_id)
                
                pose = thetas.reshape(smpl.pose_shape)  # (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
                trans = trans.reshape(smpl.trans_shape)  # np.zeros(smpl.trans_shape)

                smpl.set_params(beta=beta, pose=pose, trans=trans)

                smpl.save_to_obj(folder_name + '/{}.obj'.format(model_id))