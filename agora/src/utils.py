import os, sys, re, pickle, json
import numpy as np
import cv2
import pandas as pd

def construct_img_path(dataset_config, img_config, frame_id):

    config_list = [
        "ag",
        img_config["split"],
        img_config["vendor"],
        img_config["dataset"],
        img_config["scene"],
        img_config["num_people"],
        img_config["cam"],
        str(frame_id).zfill(5)
    ]

    config_list.pop(6) if img_config["scene"] != "archviz" else None

    img_file_name = "_".join(config_list)

    hd_path = img_file_name + "_" + img_config["resolution"] + ".png"

    return dataset_config["img_folder_path"] + "/" + hd_path, img_file_name + ".png"

def construct_mask_path(dataset_config, img_config, frame_id):

    config_list = [
        "ag",
        img_config["split"],
        img_config["vendor"],
        img_config["dataset"],
        img_config["scene"],
        img_config["num_people"],
        "mask",
        img_config["cam"],
        str(frame_id).zfill(6)
    ]

    config_list.pop(7) if img_config["scene"] != "archviz" else None

    img_file_name = "_".join(config_list)

    config_list[4] = "hdri" if img_config["scene"] == "hdri_50mm" else img_config["scene"]
    scene_config = config_list[:5]
    scene_path = "_".join(scene_config)

    mask_files_names = os.listdir(dataset_config["mask_folder_path"] + "/" + scene_path)

    mask_paths = []

    num_people = 0

    for mask_name in mask_files_names:
        if mask_name.startswith(img_file_name):

            # construct single mask path img_path + model_id + resolution + png
            hd_path = img_file_name + "_" + str(num_people).zfill(5) + "_" + img_config["resolution"] + ".png"

            mask_paths.append(
                dataset_config["mask_folder_path"] + "/" + scene_path + "/" + hd_path
            )
            num_people += 1


    return mask_paths


def read_anno(dataset_config, img_filename):
    """
    record = {
            "smpl_path" :  N
            "smplx_path" : N
            "keypoints2d" :  # Nx45x2 # convert to COCO format explicitly
            "keypoints3d" :  # Nx45x3 
            "location" :  # Nx4 (X,Y,Z,Yaw) yaw in degrees
            "cam_extrinsics" :  # 1x4 (X,Y,Z,Yaw)
        }
    """

    with open(dataset_config["cam_folder_path"] + "/" + img_filename[:-4] + ".json", "r") as annotation:
            anno = json.load(annotation)

    return anno

def get_unique_color(single_mask, all_mask):
    unique_colors = set(tuple(v) for m2d in single_mask for v in m2d)

    if len(unique_colors) == 1:
        return None, None

    for color in unique_colors:
        if color != (0, 0, 0):
            unique_color = color

    all_unique_colors = list(set(tuple(v) for m2d in all_mask for v in m2d))

    all_unique_colors = np.array(all_unique_colors, dtype="int")

    color_difference = np.absolute(all_unique_colors-unique_color)

    all_mask_color_index = np.mean(color_difference, axis=1).argmin()

    return unique_color, all_unique_colors[all_mask_color_index]

def approximate_bb(keypoints):
    x_offset = 10
    y_offset = 10

    xs = keypoints.T[0]
    ys = keypoints.T[1]

    x_min = int(xs.min()) - x_offset
    x_max = int(xs.max()) + x_offset

    y_min = int(ys.min()) - y_offset
    y_max = int(ys.max()) + y_offset

    top_left = [x_min, y_min]

    bottom_right = [x_max, y_max]

    return top_left, bottom_right

def agora2coco(joints_2d,):
    """
    coco format :
    ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder',
    'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle', 'head', 'neck', 'right_ear',
    'left_ear', 'nose', 'right_eye', 'left_eye']

    agora format:
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow","right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",

    left hand center ? , right hand center ?

    "nose", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe", "left_small_toe",
    "left_heel", "right_big_toe", "right_small_toe", "right_heel","left_thumb","left_index",
    "left_middle", "left_ring", "left_pinky", "right_thumb", "right_index", "right_middle",
    "right_ring","right_pinky",

    :param joints_2d:
    :return:
    """

    agora_coco_joint_indices = [17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 15, 12, 27, 28, 24, 25, 26]

    joints_2d_coco = joints_2d[:, agora_coco_joint_indices, :]

    return joints_2d_coco

def get_smpl(smpl_path):
    """
    example_smpl = "smpl_gt/validationset_renderpeople_adults_bfh/rp_aaron_posed_014_0_0.obj"
    smpl_path = "/home/emre/Documents/master/thesis/agora/data/ground_truth"
    get_smpl(smpl_path + "/" + example_smpl)
    """

    """
    :param smpl_folder_path:/home/tuba/Documents/emre/thesis/dataset/agora/smpl/
    :param smpl_paths: [smpl_gt/.../....obj]
    split: trainset/validationset,
    vendor: 3dpeople/axyz/humanalloy/renderpeople,
    age: adults/kids,
    dataset: bfh/body (bfh:body, hands and face. body: only body)

    :param frame_id:
    :return:

    each smpl pickle has following keys:
    ['translation', 'root_pose', 'body_pose', 'betas', 'joints', 'faces', 'vertices', 'full_pose', 'v_shaped']
    translation <class 'torch.Tensor'> torch.Size([1, 3])
    root_pose <class 'torch.Tensor'> torch.Size([1, 1, 3])
    body_pose <class 'torch.Tensor'> torch.Size([1, 23, 3])
    betas <class 'torch.Tensor'> torch.Size([1, 10])
    joints <class 'torch.Tensor'> torch.Size([1, 24, 3])
    faces <class 'numpy.ndarray'> 41328
    vertices <class 'torch.Tensor'> torch.Size([1, 6890, 3])
    full_pose <class 'torch.Tensor'> torch.Size([1, 24, 3, 3])
    v_shaped <class 'torch.Tensor'> torch.Size([1, 6890, 3])
    """


    obj_path = smpl_path[:-4] + ".pkl"

    with open(obj_path, "rb") as smpl_file:
        smpl_pickle = pickle.load(smpl_file, encoding="latin1")

    trans = smpl_pickle["translation"][0].detach().cpu().numpy()
    betas = smpl_pickle["betas"][0].detach().cpu().numpy()
    joints = smpl_pickle["joints"][0].detach().cpu().numpy()

    return trans, betas, joints

def dump_sorted(filename_list, index_list, occ_status, subset_name, scene_name="archviz", folder_name = "../selected_frames"):

    selected = zip(filename_list, index_list, occ_status)
    
    selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True) # sort by occlusion value in descending order
    os.makedirs(folder_name, exist_ok=True)

    with open(folder_name + "/" + scene_name + "/" + subset_name+".txt", "w+") as dump_file :
    
        for result in selected_sorted:
            dump_file.write(
                "{} {} #{}\n".format(result[0], result[1], result[2])
            )