import os, sys, re, pickle
import numpy as np
import cv2
import pandas as pd

from visualize import draw_keypoints_mask

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


"""def show_im(img_path, scale, show=False):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    if show:
        cv2.imshow("window_name", img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return img"""



def show_im(img_path, scale, show=False):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    if show:
        cv2.imshow("window_name", img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return img


def compare_person(all_mask, single_mask, debug=False):
    try:
        unique_colors = set(tuple(v) for m2d in single_mask for v in m2d)
        if len(unique_colors) == 1:
            return None

        for color in unique_colors:
            if color != (0, 0, 0):
                unique_color = color

        all_mask[all_mask == (0, 0, 255)] = 0  # convert bg to black

        sub_focus = np.array(all_mask, dtype=int) - unique_color  # calculate each pixel dif to instance color

        sub_focus = np.sum(np.abs(sub_focus), axis=2)  # sum over dif pixel values

        all_submask = sub_focus < 5  # assign close ones to that instance

        sub_mask = (single_mask == unique_color).all(-1)  # pixel count from instance mask

        all_pixels = np.sum(sub_mask)

        visible_pixels = np.sum(all_submask)  # count closest pixels

        ratio = visible_pixels / all_pixels

        if ratio > 1.0:
            print("Num pixels: {} / Visible Pixels: {}".format(all_pixels, visible_pixels))
            debug = True

    except Exception as e:
        print(e)
        print(unique_colors)
        sys.exit()

    if debug:
        canvas = np.ones_like(all_mask) * 255
        canvas[sub_mask] = unique_color
        canvas[all_submask] = 0
        """

        coords = np.where(single_mask == unique_color)

        xs = coords[0][0]
        ys = coords[1][0]

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        canvas = cv2.circle(canvas, (xs, ys), 10, (36, 255, 12), 3)

        canvas = cv2.putText(canvas, str(ratio), (xs, ys), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        """

        cv2.imshow("window_name", canvas)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return ratio


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

def get_smpl(smpl_folder_path, smpl_paths):
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

    betas = np.zeros((len(smpl_paths), 10))
    joints = np.zeros((len(smpl_paths), 24, 3))

    for i, smpl in enumerate(smpl_paths):
        obj_path = smpl_folder_path + smpl[:-4] + ".pkl"

        with open(obj_path, "rb") as smpl_file:
            smpl_pickle = pickle.load(smpl_file, encoding="latin1")

        betas[i] = smpl_pickle["betas"][0].detach().cpu().numpy()
        joints[i] = smpl_pickle["joints"][0].detach().cpu().numpy()

    return betas, joints


def get_joints(img_filename, scale, validation=False):
    if validation:
        joints_df = pd.read_csv("./validation_joints.csv")
    else:
        joints_df = pd.read_csv("./joints_all.csv")

    joints2d = list(joints_df[joints_df["imgPath"] == img_filename]["gt_joints_2d"])[0]
    people = joints2d.split("array(")[1:]

    joints2d_np = np.zeros((len(people), 45, 2))

    for i, person in enumerate(people):
        points = re.findall(r'\d+\.?\d*', person) #person.split(",")
        map(np.float64, points)
        points = np.array(points, dtype="float64").reshape(-1, 2)
        joints2d_np[i] = points * (720/2160)


    joints3d = list(joints_df[joints_df["imgPath"] == img_filename]["gt_joints_3d"])[0]
    people = joints3d.split("array(")[1:]

    joints3d_np = np.zeros((len(people), 45, 3))

    for i, person in enumerate(people):
        points = re.findall(r'\d+\.?\d*', person) #person.split(",")
        map(np.float64, points)
        points = np.array(points, dtype="float64").reshape(-1, 3)
        try:
            joints3d_np[i] = points * (720/2160)
        except Exception as e:
            print("joints 3d erroneous")
            print(e)
            #print(points)
            #sys.exit()


    return np.multiply(joints2d_np, scale), np.multiply(joints3d_np, scale)

def convert_joints2coco(joints_2d,):
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

def approximate_bb(pose2d, scale):
    x_offset = 10 * scale
    y_offset = 10 * scale

    xs = pose2d.T[0]
    ys = pose2d.T[1]

    x_min = int(xs.min()) - x_offset
    x_max = int(xs.max()) + x_offset

    y_min = int(ys.min()) - y_offset
    y_max = int(ys.max()) + y_offset

    top_left = [x_min, y_min]

    bottom_right = [x_max, y_max]

    return top_left, bottom_right

def crowd_metric(bbox, keypoints):

    intersection_keypoints = []

    count = 0
    for keypoint in keypoints:
        if bbox[0][0] < keypoint[0] < bbox[1][0] and bbox[0][1] < keypoint[1] < bbox[1][1]:
            count += 1
            intersection_keypoints.append([keypoint[0], keypoint[1]])

    return count, intersection_keypoints


def modified_crowd_metric(all_mask_image, person_color_code, keypoints, drop_object_occlusion=False):

    # keypoints: occluded
    # person color code: occluded

    neighbor_offset = 1
    intersection_keypoints = []

    count = 0
    for keypoint in keypoints:

        pixels = [
            [int(np.floor(keypoint[1])), int(np.floor(keypoint[0]))],
            [int(np.floor(keypoint[1])) + neighbor_offset, int(np.floor(keypoint[0]))], # bottom
            [int(np.floor(keypoint[1])), int(np.floor(keypoint[0])) + neighbor_offset], # right
            [int(np.floor(keypoint[1])) + neighbor_offset, int(np.floor(keypoint[0])) + neighbor_offset], # right - bottom
            ]

        keypoint_pixel_values = []

        for pixel in pixels:
            try:
                keypoint_pixel_values.append(all_mask_image[pixel[0], pixel[1]])
            except:
                continue

        if len(keypoint_pixel_values) == 0:
            continue

        if drop_object_occlusion and list(keypoint_pixel_values[0]) == [0,0,255]: # drop background
            continue

        """
        keypoint_pixel_values = [
            all_mask_image[int(np.floor(keypoint[1])), int(np.floor(keypoint[0]))],
            all_mask_image[int(np.floor(keypoint[1])) + neighbor_offset, int(np.floor(keypoint[0]))], # bottom
            all_mask_image[int(np.floor(keypoint[1])), int(np.floor(keypoint[0])) + neighbor_offset], # right
            all_mask_image[int(np.floor(keypoint[1])) + neighbor_offset, int(np.floor(keypoint[0])) + neighbor_offset], # right - bottom
            ]
        """

        if np.mean(np.absolute(keypoint_pixel_values - person_color_code), axis=1).all() != 0.0 :  # if the pixel value isn't the same
            count += 1
            intersection_keypoints.append([keypoint[0], keypoint[1]])
            """
            print(keypoint)
            print(keypoint_pixel_values)
            print(person_color_code)
            draw_keypoints_mask(all_mask_image, np.expand_dims(keypoint, axis=(0, 1)), 1)
            """

    return count, intersection_keypoints


def get_cam(cam_path, pkl_id=None, with_joints=False, validation=False):
    """
    :param cam_path:
    :param pkl_id:
    :return: pandas DF
        columns:
        Index(['X', 'Y', 'Z', 'Yaw', 'imgPath', 'camYaw', 'camZ', 'camY', 'camX',
           'gender', 'gt_path_smpl', 'gt_path_smplx', 'kid', 'occlusion',
           'isValid', 'age', 'ethnicity'],
          dtype='object')

        check readme for details

        ['X', 'Y', 'Z', 'Yaw', 'imgPath', 'camYaw', 'camZ', 'camY', 'camX',
       'gender', 'gt_path_smpl', 'gt_path_smplx', 'kid', 'occlusion',
       'isValid', 'age', 'ethnicity', 'gt_joints_2d', 'gt_joints_3d',
       'gt_verts']
    """
    if validation:
        print("processing validation smpl")
        appended_data = []
        for pkl_id in range(10):

            path = cam_path + "/validation_SMPL/SMPL/validation_" + str(pkl_id) + "_withjv" + ".pkl"

            with open(path, "rb") as cam_file:
                df = pickle.load(cam_file, encoding="latin1")

                appended_data.append(df)

        appended_data = pd.concat(appended_data)


        joints_data = appended_data[["imgPath", "gt_path_smpl", "gt_joints_2d", "gt_joints_3d"]]
        joints_data.to_csv("validation_joints.csv")
        print("saved to csv")

        return appended_data

    if pkl_id is not None:
        print("processing pickle number ", pkl_id)

        if with_joints:
            path = cam_path + "/train_SMPL/SMPL/train_" + str(pkl_id) + "_withjv" + ".pkl"
        else:
            path = cam_path + "/train_Cam/Cam/train_" + str(pkl_id) + ".pkl"

        with open(path, "rb") as cam_file:
            df = pickle.load(cam_file, encoding="latin1")

        if with_joints:
            joints_data = df[["imgPath", "gt_path_smpl", "gt_joints_2d", "gt_joints_3d"]]
            joints_data.to_csv("joints_{}.csv".format(pkl_id))
            print("saved to csv")

        return df

    else:
        #### o/w specified only this part runs
        appended_data = []
        for pkl_id in range(10):
            if with_joints:
                path = cam_path + "/train_SMPL/SMPL/train_" + str(pkl_id) + "_withjv" + ".pkl"
            else:
                path = cam_path + "/train_Cam/Cam/train_" + str(pkl_id) + ".pkl"

            with open(path, "rb") as cam_file:
                df = pickle.load(cam_file, encoding="latin1")

                appended_data.append(df)

        appended_data = pd.concat(appended_data)
        #####

        if with_joints:
            joints_data = appended_data[["imgPath", "gt_path_smpl", "gt_joints_2d", "gt_joints_3d"]]
            joints_data.to_csv("joints.csv")
            print("saved to csv")

        return appended_data


##### TEK SEFERLIK

def merge_joint_data():
    appended_data = []
    for i in range(10):
        df = pd.read_csv("./joints_{}.csv".format(i))
        appended_data.append(df)
    appended_data = pd.concat(appended_data)

    appended_data.to_csv("joints_all.csv")

    convert_joint_column_type()

    return

def convert_joint_column_type():
    joints_df = pd.read_csv("./joints_all.csv")

    for index, row in joints_df.iterrows():
        joints2d = row["gt_joints_2d"]
        people = joints2d.split("array(")[1:]

        joints2d_list = []

        for i, person in enumerate(people):
            points = re.findall(r'\d+\.?\d*', person)  # person.split(",")
            map(float, points)
            points = np.array(points).reshape(-1, 2)
            joints2d_list.append(points)

        joints_df.at[index, "gt_joints_2d"] = joints2d_list

        joints3d = row["gt_joints_3d"]
        people = joints3d.split("array(")[1:]

        joints3d_list = []

        for i, person in enumerate(people):
            points = re.findall(r'\d+\.?\d*', person)  # person.split(",")
            map(float, points)
            points = np.array(points).reshape(-1, 3)
            joints3d_list.append(points)

        joints_df.at[index, "gt_joints_3d"] = joints3d_list

    joints_df.to_csv("joints_processed.csv")

    return joints_df