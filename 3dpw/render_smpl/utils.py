import trimesh
import numpy as np
import cv2
import os, time, sys, colorsys
import json
from matplotlib import cm as mpl_cm, colors as mpl_colors


color_list = [
    [255, 0, 0], # red
    [0, 0, 255], # blue
    [.7, .7, .9] * 255, # pink
    [.9, .9, .8] * 255, # neutral
    [.7, .75, .5] * 255, # capsule
    [.5, .7, .75] * 255 # yellow
]

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

distinct_colors_new = distinct_colors.copy()
distinct_colors_new[13] = distinct_colors[20]
distinct_colors_new[20] = distinct_colors[13]
distinct_colors = distinct_colors_new

#cm = mpl_cm.get_cmap('jet')
#norm_gt = mpl_colors.Normalize()
#distinct_colors = cm(norm_gt(list(range(24))))[:, :-1].tolist()

"""
distinct_colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (255, 0, 255), # Magenta
    (0, 255, 255), # Cyan
    (128, 0, 0),   # Maroon
    (0, 128, 0),   # Green (128)
    (0, 0, 128),   # Navy
    (128, 128, 0), # Olive
    (128, 0, 128), # Purple
    (0, 128, 128), # Teal
    (192, 192, 192), # Silver
    (128, 128, 128), # Gray
    (255, 165, 0),  # Orange
    (255, 192, 203), # Pink
    (165, 42, 42),  # Brown
    (0, 128, 128),  # Teal (128)
    (210, 105, 30), # Chocolate
    (139, 69, 19),  # SaddleBrown
    (0, 255, 127),  # SpringGreen
    (255, 69, 0),   # Red-Orange
    (0, 139, 139),  # DarkCyan
    (255, 20, 147)  # DeepPink
]
"""

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

def count_pixels(mask):
    
    pixels = mask.reshape(-1, 3)

    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    color_counts = {}
    for color, count in zip(unique_colors, counts):
        if np.sum(color) == 0:
            continue

        color_counts[tuple(color)] = count

    return color_counts

def regional_occlusion(all_mask_counter, individual_masks_counter):

    occlusion_ratios = []

    for mask_counter in individual_masks_counter:
        unique_color = list(mask_counter.keys())[0]

        total_pixels = mask_counter[unique_color]

        visible_pixels = all_mask_counter[unique_color]

        regional_occlusion = (1 - (visible_pixels / total_pixels)) * 100

        occlusion_ratios.append(regional_occlusion)

    return occlusion_ratios


def load_mesh(path_list):
    vertices_list = []
    faces_list = []

    for path in path_list:
        mesh_obj = trimesh.load_mesh(path)
        vertices_list.append(mesh_obj.vertices)
        faces_list.append(mesh_obj.faces)
    
    return vertices_list, faces_list

def load_cam(cam_path):

    with open(cam_path, "r") as cam_json:
        cam_data = json.load(cam_json)

    intrinsics = np.array(cam_data["intrinsics"])
    extrinsics = np.array(cam_data["extrinsics"])

    return intrinsics, extrinsics

def project_points(vertices_list, intrinsics, extrinsics):
    image_points_list = []

    for vertices in vertices_list:
        rotation_matrix = extrinsics[:3, :3]  # 3x3 rotation matrix
        translation_vector = extrinsics[:3, -1]  # 3x1 translation vector

        projection_matrix = np.dot(intrinsics, np.hstack((rotation_matrix, translation_vector.reshape(-1, 1))))
        projected_vertices = np.dot(projection_matrix, np.hstack((vertices, np.ones((len(vertices), 1)))).T).T # move to homogeneous coords
        image_points = projected_vertices[:, :2] / projected_vertices[:, 2].reshape(-1, 1) # get back from homogeneous coords.
        image_points_list.append(image_points)

    return image_points_list


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