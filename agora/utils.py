import os, sys, re, pickle, json
import numpy as np
import cv2, math, colorsys, trimesh
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
    y_offset = 50

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

def dump_sorted(filename_list, index_list, occ_status, subset_name, scene_name="archviz", folder_name = "./agora/selected_frames"):

    selected = zip(filename_list, index_list, occ_status)
    
    selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True) # sort by occlusion value in descending order
    os.makedirs(folder_name + "/" + scene_name, exist_ok=True)

    with open(folder_name + "/" + scene_name + "/" + subset_name+".txt", "w+") as dump_file :
    
        for result in selected_sorted:
            dump_file.write(
                "{} {} #{}\n".format(result[0], result[1], result[2])
            )

def project_points(vertices_list, intrinsics, extrinsics):
    image_points_list = []

    for vertices in vertices_list:
        rotation_matrix = extrinsics[:3, :3]  # 3x3 rotation matrix
        translation_vector = extrinsics[:3, -1]  # 3x1 translation vector

        projection_matrix = np.dot(intrinsics, np.hstack((rotation_matrix, translation_vector.reshape(-1, 1))))
        projected_vertices = np.dot(projection_matrix, np.hstack((vertices, np.ones((len(vertices), 1)))).T).T # move to homogeneous coords
        image_points = projected_vertices[:, :2] / projected_vertices[:, 2].reshape(-1, 1) # get back from homogeneous coords.
        image_points *= (720/2160)
        
        image_points_list.append(image_points)

    return image_points_list

def rotate_points_3d(points, angle_degrees):
    """
    Rotate a list of 3D points around the Z-axis.

    Parameters:
        points (numpy.ndarray): An Nx3 array representing N points with (x, y, z) coordinates.
        angle_degrees (float): The angle of rotation in degrees.

    Returns:
        numpy.ndarray: An Nx3 array with the rotated points.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix for Z-axis
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])

    # Apply rotation to all points
    rotated_points = np.dot(points, rotation_matrix)

    return rotated_points

def construct_extrinsic_matrix(parameters):
    """
    Construct the extrinsic parameter matrix for a 3D transformation.

    Parameters:
        parameters (list): List of 4 parameters [x_translation, y_translation, z_translation, z_rotation_degrees].

    Returns:
        numpy.ndarray: The 4x4 extrinsic parameter matrix.
    """
    x_translation, y_translation, z_translation, z_rotation_degrees = parameters

    # Convert rotation angle to radians
    z_rotation_radians = np.radians(z_rotation_degrees)

    # Translation matrix
    translation_matrix = np.array([
        [1, 0, 0, x_translation],
        [0, 1, 0, y_translation],
        [0, 0, 1, z_translation],
        [0, 0, 0, 1]
    ])

    # Rotation matrix for Z-axis
    rotation_matrix = np.array([
        [np.cos(z_rotation_radians), -np.sin(z_rotation_radians), 0, 0],
        [np.sin(z_rotation_radians), np.cos(z_rotation_radians), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Combine translation and rotation matrices
    extrinsic_matrix = np.dot(translation_matrix, rotation_matrix)

    return extrinsic_matrix


def load_mesh(path_list):
    vertices_list = []
    faces_list = []

    for path in path_list:
        mesh_obj = trimesh.load_mesh(path)
        vertices_list.append(mesh_obj.vertices)
        faces_list.append(mesh_obj.faces)
    
    return vertices_list, faces_list

def vertex_classes(part_segm_filepath):
    part_segm = json.load(open(part_segm_filepath))
    _, faces_list = load_mesh(["./3dpw/render_smpl/example_smpl.obj"])
    faces_list = faces_list[0]

    vertex_class_list = np.zeros(6890)
    vertex_class_names = {}
    faces_per_class = {}

    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_class_list[v] = part_idx
        vertex_class_names[part_idx] = k
        faces_per_class[part_idx] = []

    face_classes = []

    for face_id, face in enumerate(faces_list):

        current_vertex_classes = [vertex_class_list[face[0]], vertex_class_list[face[1]], vertex_class_list[face[2]]]

        pixel_class_counts = np.bincount(current_vertex_classes)

        predicted_class_for_pixel = np.argmax(pixel_class_counts)

        face_classes.append(predicted_class_for_pixel)
        
        faces_per_class[predicted_class_for_pixel].append(face_id)


    return face_classes, vertex_class_names, faces_per_class

def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Distribute hues evenly
        saturation = 0.7  # You can adjust this value
        lightness = 0.6  # You can adjust this value
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Convert the float values to integers in the range [0, 255]
        r, g, b = [int(x * 255) for x in rgb]
        colors.append((r, g, b))
    return colors

distinct_colors = generate_distinct_colors(24)

def count_pixels(mask):
    
    pixels = mask.reshape(-1, 3)

    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    color_counts = {}
    for color, count in zip(unique_colors, counts):
        if np.sum(color) == 0:
            continue

        color_counts[tuple(color)] = count

    return color_counts

def add_padding(image, left_padding, right_padding):
    # Get the height and width of the original image
    height, width = image.shape[:2]
    
    # Calculate the new width with padding
    new_width = width + left_padding + right_padding
    
    # Create a new white image with the desired width and the same height as the original image
    padded_image = np.full((height, new_width, image.shape[2]), 255, dtype=image.dtype)  # 255 represents white

    # Copy the original image to the center of the new image
    padded_image[:, left_padding:left_padding + width] = image

    return padded_image