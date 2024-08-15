import trimesh
import numpy as np
import cv2
import os, time, sys, colorsys
import json
from utils import *
import random

save_vis = False
visualize = False
self_occlusion = False
verbose = False

random.seed(42)
distinct_colors_new = distinct_colors.copy()
random.shuffle(distinct_colors_new)

def render(scene_silhouette_canvas, z_buffer, faces, image_points, vertices, color, face_classes):
    
    image_height, image_width = scene_silhouette_canvas.shape[:2]

    model_silhouette_canvas = np.zeros_like(scene_silhouette_canvas)

    model_full_segment_canvas = np.zeros_like(scene_silhouette_canvas)

    model_z_buffer = np.ones_like(z_buffer) * -np.inf # for body segmentation

    num_joints = len(np.unique(face_classes))
    pixel_counter = np.zeros(num_joints, dtype=int)
    truncation_counter = np.zeros(num_joints, dtype=int)


    # Draw triangles on the image canvas with proper depth testing
    for face_id, face in enumerate(faces):

        triangle = np.array([image_points[face[0]], image_points[face[1]], image_points[face[2]]], dtype=np.int32)
        
        # Calculate the bounding box of the triangle
        """
        min_x = max(min(triangle[:, 0]), 0)
        max_x = min(max(triangle[:, 0]), image_width - 1)
        min_y = max(min(triangle[:, 1]), 0)
        max_y = min(max(triangle[:, 1]), image_height - 1)
        """

        min_x = min(triangle[:, 0])
        max_x = max(triangle[:, 0])
        min_y = min(triangle[:, 1])
        max_y = max(triangle[:, 1])
        
        # Iterate over the bounding box and perform depth testing
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                pixel_counter[int(face_classes[face_id])] += 1

                if (image_height <= y) or (y < 0) or (image_width <= x) or (x < 0):
                    truncation_counter[int(face_classes[face_id])] += 1
                    continue

                inside_test = cv2.pointPolygonTest(triangle, (int(x), int(y)), measureDist=True)

                if inside_test >= 0:
                    model_silhouette_canvas[y, x] = color 
                    
                    barycentric_triangle = np.hstack((triangle, np.ones((len(triangle), 1)))).T # move to homogeneous coords and convert to form below
                    # ax bx cx
                    # ay by cy
                    #  1  1  1
                    
                    # calculate barycentric coords
                    (alpha, beta, gamma) = np.linalg.lstsq(barycentric_triangle, (int(x), int(y), 1), rcond=None)[0] # point in homogeneous coords

                    depth = vertices[face[0], 2] * alpha + \
                            vertices[face[1], 2] * beta + \
                            vertices[face[2], 2] * gamma
                    

                    
                    if depth > z_buffer[y, x]: # greater depth means closer to camera, since camera looking towards -z axis
                        z_buffer[y, x] = depth
                        scene_silhouette_canvas[y, x] = color

                    if depth > model_z_buffer[y, x]:
                        model_z_buffer[y, x] = depth

                        model_full_segment_canvas[y, x] = distinct_colors_new[int(face_classes[face_id])]


    truncation_stats = np.zeros_like(truncation_counter, dtype=np.float32)

    for joint_id in range(num_joints):
        if pixel_counter[joint_id] == 0:
            truncation_stats[joint_id] = 100
        else:
            truncation_stats[joint_id] = truncation_counter[joint_id] / pixel_counter[joint_id]
            truncation_stats[joint_id] *= 100 


    return scene_silhouette_canvas, z_buffer, model_silhouette_canvas, model_full_segment_canvas, truncation_stats

def render_segment(segment_canvas, faces, image_points, vertices, color, face_classes, sub_face_list):
    
    image_height, image_width = segment_canvas.shape[:2]
    
    truncation_counter = 0

    # Draw triangles on the image canvas with proper depth testing
    for face_id, face in enumerate(faces):
        if face_id in sub_face_list:
            pass
        else:
            continue

        triangle = np.array([image_points[face[0]], image_points[face[1]], image_points[face[2]]], dtype=np.int32)
        
        # Calculate the bounding box of the triangle
        min_x = max(min(triangle[:, 0]), 0)
        max_x = min(max(triangle[:, 0]), image_width - 1)
        min_y = max(min(triangle[:, 1]), 0)
        max_y = min(max(triangle[:, 1]), image_height - 1)
        
        # Iterate over the bounding box and perform depth testing
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if not image_height > y >= 0 or not image_width > x >= 0:
                    truncation_counter += 1
                    continue

                inside_test = cv2.pointPolygonTest(triangle, (int(x), int(y)), measureDist=True)

                if inside_test >= 0:
                    segment_canvas[y, x] = distinct_colors_new[int(face_classes[face_id])]                         

    return segment_canvas, truncation_counter

def main():

    smpl_obj_path = "/home/tuba/Documents/emre/thesis/occlusionIndex/3dpw/smpl_objects"
    images_path = "/home/tuba/Documents/emre/thesis/dataset/3dpw/imageFiles"
    part_segm_filepath = "/home/tuba/Documents/emre/thesis/occlusionIndex/smpl_vert_segmentation.json"
    
    seq_list = ['courtyard_dancing_01', 'courtyard_goodNews_00',
                'courtyard_giveDirections_00', 'courtyard_hug_00', 
                'courtyard_dancing_00', 'courtyard_shakeHands_00', 
                'downtown_bar_00','courtyard_warmWelcome_00', 
                'courtyard_captureSelfies_00', 'courtyard_basketball_00']
        
    #seq_list = ['courtyard_warmWelcome_00'] # TODO remove
    
    face_classes, vertex_class_names, faces_per_class = vertex_classes(part_segm_filepath=part_segm_filepath)
    num_joints = len(faces_per_class.keys())
    
    #seq_list = ["courtyard_dancing_01"] #cheating to process only one image
    
    for seq_name in seq_list:
        
        print("Processing {}...".format(seq_name))
        
        seq_path = smpl_obj_path + "/" + seq_name

        image_list = os.listdir(seq_path)
        image_list.sort()

        image_list = ["image_00045"] #cheating to process only one image
        
        duration = []

        for frame_id, image_name in enumerate(image_list):
            start_time = time.time()
            remaining_secs = np.mean(duration)*(len(image_list)-frame_id)
            print("%{:.2f} Processing {}/{}... ETA: {:.0f}mins {:.0f}secs".format(frame_id*100/len(image_list), seq_name, image_name, remaining_secs//60, remaining_secs%60))
            
            ##############################################################################################################################################
            # load image and annotations
            image = cv2.imread(images_path + "/" + seq_name + "/" + image_name + ".jpg")
            
            image_height, image_width = image.shape[:2]

            # collect mesh paths
            image_folder_path = seq_path + "/" + image_name
            
            mesh_path_list = []

            cam_path = None

            for mesh_file in os.listdir(image_folder_path):
                if mesh_file.endswith(".obj"):
                    mesh_path_list.append(image_folder_path + "/" + mesh_file)
                else:
                    cam_path = image_folder_path + "/" + mesh_file
            num_model = len(mesh_path_list)
            
            ##############################################################################################################################################
            # load mesh object
            vertices_list, faces_list = load_mesh(mesh_path_list)

            # load cam params
            intrinsics, extrinsics = load_cam(cam_path)

            # project triangles to image plane
            image_points_list = project_points(vertices_list=vertices_list, intrinsics=intrinsics, extrinsics=extrinsics)

            ##############################################################################################################################################
            # initialize canvas and z-buffer
            scene_silhouette_canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            model_silhouette_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)
            model_occluded_silhouette_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)
            
            # body segments
            model_full_segment_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)
            model_occluded_segment_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)
            model_visible_segment_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)

            # per segment 
            per_segment_visible_canvas = np.zeros((num_model, num_joints, image_height, image_width, 3 ), dtype=np.uint8)

            
            # z-buffer for scene silhouettes
            z_buffer = np.ones((image_height, image_width), dtype=np.float64) * -np.inf

            # detect truncations
            truncations = np.zeros((num_model, num_joints)) 

            ##############################################################################################################################################
            if verbose:
                silhouette_render_start = time.time()

            # render meshes with z buffer
            for model_id in range(num_model):
                
                scene_silhouette_canvas, z_buffer, \
                model_silhouette, model_full_segment, truncation_counter = render(scene_silhouette_canvas=scene_silhouette_canvas,
                                                                            z_buffer=z_buffer,
                                                                            faces=faces_list[model_id], 
                                                                            image_points=image_points_list[model_id], 
                                                                            vertices=vertices_list[model_id], 
                                                                            color=color_list[model_id], 
                                                                            face_classes=face_classes)
                
                model_silhouette_canvas[model_id] = model_silhouette
                model_full_segment_canvas[model_id] = model_full_segment
                truncations[model_id] = truncation_counter

            if verbose:
                body_segment_render_start = time.time()
                print("Silhouette rendering took {} secs".format(body_segment_render_start - silhouette_render_start))

            # mask occluded body segments
            for model_id in range(num_model):
                color_difference = np.abs(scene_silhouette_canvas - color_list[model_id])

                # Calculate the sum of color differences along the color channels (R, G, B)
                color_difference_sum = color_difference.sum(axis=-1)

                # Create a mask for pixels that are not the exclude_color
                mask = color_difference_sum == 0

                # Apply the mask to select the pixels that are not the exclude_color
                model_occluded_segment_canvas[model_id] = model_full_segment_canvas[model_id].copy()
                model_occluded_segment_canvas[model_id][mask] = 0

                model_occluded_silhouette_canvas[model_id] = model_silhouette_canvas[model_id].copy()
                model_occluded_silhouette_canvas[model_id][mask] = 0

                model_visible_segment_canvas[model_id] = model_full_segment_canvas[model_id].copy()
                model_visible_segment_canvas[model_id][~mask] = 0

                if self_occlusion:

                    if verbose:
                        per_segment_render_start = time.time()

                    # render per body segment
                    for joint_id in vertex_class_names.keys():
                        segment_silhouette_canvas = np.zeros_like(scene_silhouette_canvas)
                        segment_silhouette_canvas, truncation_counter = render_segment(segment_canvas=segment_silhouette_canvas,
                                                                        faces=faces_list[model_id], 
                                                                        image_points=image_points_list[model_id], 
                                                                        vertices=vertices_list[model_id], 
                                                                        color=color_list[model_id], 
                                                                        face_classes=face_classes,
                                                                        sub_face_list=faces_per_class[joint_id])
                        
                        per_segment_visible_canvas[model_id][joint_id] = segment_silhouette_canvas
                        
                        per_segment_counter = count_pixels(mask=per_segment_visible_canvas[model_id][joint_id])

                        if len(per_segment_counter.keys()) == 0: # if the segment is fully truncated, leave it as empty canvas
                            continue

                        unique_color = list(per_segment_counter.keys())[0]

                        total_pixels = per_segment_counter[unique_color]
                        
                        # mask out occluded pixels 
                        per_segment_visible_canvas[model_id][joint_id][~mask] = 0

                        per_segment_counter = count_pixels(mask=per_segment_visible_canvas[model_id][joint_id])

                        if not per_segment_counter: # if empty dict -> the segment is fully occluded by other people
                            continue

                        visible_pixels = per_segment_counter[unique_color]

                        visibility_rate = (visible_pixels / total_pixels) * 100

                        # if the remaining only visible by 10%, assume it is occluded.
                        if visibility_rate < 10:
                            #print("{} has visibility of less than 10%".format(vertex_class_names[joint_id]))
                            per_segment_visible_canvas[model_id][joint_id] = np.zeros_like(per_segment_visible_canvas[model_id][joint_id])
                        if verbose:
                            per_segment_render_end = time.time()
                    if verbose:
                        print("Rendering per segment for model {} took {} secs".format(model_id, per_segment_render_end-per_segment_render_start))
            if verbose:
                body_segment_render_end = time.time()
                print("Rendering body segment took {} secs".format(body_segment_render_end-body_segment_render_start))     

            ##############################################################################################################################################
            # calculate statictics

            frame_occlusion_statistics = {}

            scene_silhouette_pixel_counter = count_pixels(mask=scene_silhouette_canvas)

            model_silhouette_pixel_counter = []
            for model_id in range(num_model):
                
                ############################################################################################
                if verbose:
                    silhouette_calculation_start = time.time()

                # calculate overall occlusion
                model_silhouette_pixel_counter = count_pixels(mask=model_silhouette_canvas[model_id])

                if len(model_silhouette_pixel_counter.keys()) == 0: # if the body is fully truncated, leave it as empty canvas
                    model_regional_occlusion = 0
                else:
                    unique_color = list(model_silhouette_pixel_counter.keys())[0]

                    total_pixels = model_silhouette_pixel_counter[unique_color]

                    visible_pixels = scene_silhouette_pixel_counter[unique_color]

                    model_regional_occlusion = (1 - (visible_pixels / total_pixels)) * 100

                model_occlusion_statistics = {
                    "occlusion": {
                        "overall": model_regional_occlusion,
                    },
                    "self_occlusion": {
                        "overall": 0.0
                    },
                    "truncation": {
                        "overall": 0.0
                    }
                }
                # set as 0 as default, for the visible joints
                for joint_id, joint_name in enumerate(vertex_class_names.values()):
                    model_occlusion_statistics["occlusion"][joint_name] = 0.0
                    model_occlusion_statistics["self_occlusion"][joint_name] = 0.0
                    model_occlusion_statistics["truncation"][joint_name] = truncations[model_id][joint_id]

                model_occlusion_statistics["truncation"]["overall"] = np.mean(truncations[model_id])

                ############################################################################################
                if verbose:
                    body_segment_calculation_start = time.time()
                    print("Silhouette calculations took {} secs".format(body_segment_calculation_start - silhouette_calculation_start))

                # calculate occlusion per body segment
                model_full_segment_counter = count_pixels(mask=model_full_segment_canvas[model_id])
                model_occluded_segment_counter = count_pixels(mask=model_occluded_segment_canvas[model_id])

                for body_segment_color in list(model_occluded_segment_counter.keys()):
                    total_pixels = model_full_segment_counter[body_segment_color]

                    occluded_pixels = model_occluded_segment_counter[body_segment_color]

                    body_segment_occlusion_rate = (occluded_pixels / total_pixels) * 100

                    joint_name = vertex_class_names[distinct_colors.index(body_segment_color)]
                    
                    model_occlusion_statistics["occlusion"][joint_name] = body_segment_occlusion_rate

                ############################################################################################
                if verbose:
                    self_occlusion_calculation_start = time.time()
                    print("Body segment calculations took {} secs".format(self_occlusion_calculation_start - body_segment_calculation_start))

                if self_occlusion:
                    # calculate self-occlusion
                    model_visible_segment_counter = count_pixels(mask=model_visible_segment_canvas[model_id])
                    per_segment_self_occlusion_list = []

                    for joint_id in vertex_class_names.keys():
                        
                        per_segment_counter = count_pixels(mask=per_segment_visible_canvas[model_id][joint_id])

                        if not per_segment_counter: # if empty dict -> the segment is fully occluded by other people, or truncated
                            model_occlusion_statistics["self_occlusion"][vertex_class_names[joint_id]] = 0.0
                            continue

                        unique_color = list(per_segment_counter.keys())[0]

                        total_pixels = per_segment_counter[unique_color]

                        if unique_color in model_visible_segment_counter.keys():
                            visible_pixels = model_visible_segment_counter[unique_color]
                        else:
                            visible_pixels = 0
                        
                        
                        
                        segment_regional_occlusion = (1 - (visible_pixels / total_pixels)) * 100

                        """
                        segment_ = np.concatenate([per_segment_visible_canvas[model_id][joint_id], model_visible_segment_canvas[model_id]], axis=1)
                        
                        
                        cv2.imshow(str(segment_regional_occlusion) + " {}".format(vertex_class_names[joint_id]), cv2.resize(segment_, (image_width*2//2, image_height//2)))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        """

                        model_occlusion_statistics["self_occlusion"][vertex_class_names[joint_id]] = segment_regional_occlusion
                        per_segment_self_occlusion_list.append(segment_regional_occlusion)
                    
                    model_occlusion_statistics["self_occlusion"]["overall"] = np.mean(per_segment_self_occlusion_list)
                    if verbose:
                        self_occlusion_calculation_end = time.time()
                        print("Per segment calculations took {} secs".format(self_occlusion_calculation_end - self_occlusion_calculation_start))

                ############################################################################################

                frame_occlusion_statistics[model_id] = model_occlusion_statistics

            ##############################################################################################################################################
            """
            my_list = (
            add_padding(image=image, left_padding=5, right_padding=5),
            add_padding(image=scene_silhouette_canvas, left_padding=5, right_padding=5),
            #add_padding(image=model_visible_segment_canvas[0], left_padding=5, right_padding=5),
            #add_padding(image=model_silhouette_canvas[0], left_padding=5, right_padding=5),
            #add_padding(image=model_occluded_silhouette_canvas[0], left_padding=5, right_padding=5),
            )
            created_img = np.concatenate(my_list, axis=1)
            created_img = cv2.resize(created_img, (image_width*len(my_list)//2, image_height//2))

            cv2.imshow('testcase', created_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("./3dpw/experiment_figures/{}_exp1_visualized.png".format(image_name), created_img)            
            sys.exit()
            """

            # visualize silhouettes
            if save_vis:
                divider = 2
                
                silhouette_canvas_list = []

                # show results
                scene_silhouette_canvas = cv2.resize(scene_silhouette_canvas, (image_width//divider, image_height//divider))
                scene_silhouette_canvas = add_padding(image=scene_silhouette_canvas, left_padding=5, right_padding=5)
                
                silhouette_canvas_list.append(scene_silhouette_canvas)

                for model_id in range(num_model):
                    model_silhouette = cv2.resize(model_silhouette_canvas[model_id], (image_width//divider, image_height//divider))
                    model_silhouette = add_padding(image=model_silhouette, left_padding=5, right_padding=5)

                    silhouette_canvas_list.append(model_silhouette)

                silhouette_concat_img = np.concatenate(tuple(silhouette_canvas_list), axis=1)
                
                if visualize:
                    cv2.imshow('silhouette concat image', silhouette_concat_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            ############################################################################################
            if verbose:
                visualization_start = time.time()
            
            if save_vis:
                # visualize body segments
                segment_canvas_list = []

                # show results
                for model_id in range(num_model):
                    model_full_segment = cv2.resize(model_full_segment_canvas[model_id], (image_width//divider, image_height//divider))
                    model_occluded_segment = cv2.resize(model_occluded_segment_canvas[model_id], (image_width//divider, image_height//divider))

                    cv2.putText(model_occluded_segment, "Occlusion (%)", (image_width//divider - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    for i, joint_name in enumerate(frame_occlusion_statistics[model_id]["occlusion"].keys()):
                        segment_occlusion_ratio = frame_occlusion_statistics[model_id]["occlusion"][joint_name]
                        cv2.putText(model_occluded_segment, "{}: {}".format(joint_name, round(segment_occlusion_ratio)), (image_width//divider - 180, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        #print("{}: {}".format(joint_name, round(segment_occlusion_ratio)))
                    
                    model_full_segment = add_padding(image=model_full_segment, left_padding=5, right_padding=5)
                    model_occluded_segment = add_padding(image=model_occluded_segment, left_padding=5, right_padding=5)

                    segment_canvas_list.append(model_full_segment)
                    segment_canvas_list.append(model_occluded_segment)

                segment_concat_img = np.concatenate(tuple(segment_canvas_list), axis=1)
                
                if visualize:
                    cv2.imshow('segment concat image', segment_concat_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            ############################################################################################
            """
            # visualize single body segments
            per_segment_canvas_list = []

            # show results
            for model_id in range(num_model):
                for joint_id in vertex_class_names.keys():
                    per_segment_visible = cv2.resize(per_segment_visible_canvas[model_id][joint_id], (image_width//divider, image_height//divider))
                    joint_name = vertex_class_names[joint_id]
                    cv2.putText(per_segment_visible, "{:.2f}".format(frame_occlusion_statistics[0]["self_occlusion"][joint_name]), (image_width//divider - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    if visualize:
                        cv2.imshow('per segment image {}'.format(joint_name), per_segment_visible)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
            """
            ############################################################################################
            if self_occlusion and save_vis:
                # visualize body segments
                per_segment_canvas_list = []

                # show results
                for model_id in range(num_model):
                    model_full_segment = cv2.resize(model_full_segment_canvas[model_id], (image_width//divider, image_height//divider))
                    model_visible_segment = cv2.resize(model_visible_segment_canvas[model_id], (image_width//divider, image_height//divider))

                    cv2.putText(model_visible_segment, "Occlusion (%)", (image_width//divider - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    for i, joint_name in enumerate(frame_occlusion_statistics[model_id]["self_occlusion"].keys()):
                        segment_self_occlusion_ratio = frame_occlusion_statistics[model_id]["self_occlusion"][joint_name]
                        cv2.putText(model_visible_segment, "{}: {}".format(joint_name, round(segment_self_occlusion_ratio)), (image_width//divider - 180, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    model_full_segment = add_padding(image=model_full_segment, left_padding=5, right_padding=5)
                    model_visible_segment = add_padding(image=model_visible_segment, left_padding=5, right_padding=5)

                    per_segment_canvas_list.append(model_full_segment)
                    per_segment_canvas_list.append(model_visible_segment)

                per_segment_concat_img = np.concatenate(tuple(per_segment_canvas_list), axis=1)
                
                if visualize:
                    cv2.imshow('per_segment concat image', per_segment_concat_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if verbose:
                visualization_end = time.time()
                print("Visualization took {}. secs".format(visualization_end - visualization_start))
            
            ##############################################################################################################################################
            # save
            save_path = "./3dpw/regional_occlusion/{}".format(seq_name)
            os.makedirs(save_path, exist_ok=True)

            if save_vis:
                # save generated mask

                cv2.imwrite("{}/{}.jpg".format(save_path, image_name), scene_silhouette_canvas)
                cv2.imwrite("{}/{}_silhouettes.jpg".format(save_path, image_name), silhouette_concat_img)
                cv2.imwrite("{}/{}_segments.jpg".format(save_path, image_name), segment_concat_img)
                if self_occlusion:
                    cv2.imwrite("{}/{}_self_occlusion.jpg".format(save_path, image_name), per_segment_concat_img)
                
            with open("{}/{}.json".format(save_path, image_name), "w+") as occlusion_filepointer:
                json.dump(frame_occlusion_statistics, occlusion_filepointer)

            end_time = time.time()
            duration.append(end_time-start_time)
            sys.exit()


if __name__ == "__main__":
    main()