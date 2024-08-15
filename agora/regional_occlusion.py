import os, sys, pickle, time, math
from visualize import *
from utils import *
import trimesh
from occlusion_tools import occlusion_index, filter_by_criterion, form_criterion, coco_subset
from agora_projection import project2d


save_vis = True
visualize = False
self_occlusion = False
verbose = False

color_list = generate_distinct_colors(15)

def render(scene_silhouette_canvas, z_buffer, faces, image_points, vertices, color, face_classes):
    
    image_height, image_width = scene_silhouette_canvas.shape[:2]

    model_silhouette_canvas = np.zeros_like(scene_silhouette_canvas)

    model_full_segment_canvas = np.zeros_like(scene_silhouette_canvas)

    model_z_buffer = np.ones_like(z_buffer) * np.inf # for body segmentation

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
                    

                    
                    if depth < z_buffer[y, x]: # smaller depth means closer to camera, since camera looking towards +z axis
                        z_buffer[y, x] = depth
                        scene_silhouette_canvas[y, x] = color

                    if depth < model_z_buffer[y, x]:
                        model_z_buffer[y, x] = depth

                        model_full_segment_canvas[y, x] = distinct_colors[int(face_classes[face_id])]


    truncation_stats = np.zeros_like(truncation_counter, dtype=np.float32)

    for joint_id in range(num_joints):
        if pixel_counter[joint_id] == 0:
            truncation_stats[joint_id] = 100
        else:
            truncation_stats[joint_id] = truncation_counter[joint_id] / pixel_counter[joint_id]
            truncation_stats[joint_id] *= 100 


    return scene_silhouette_canvas, z_buffer, model_silhouette_canvas, model_full_segment_canvas, truncation_stats



if __name__ == '__main__':

    dataset_config = {
        "img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/imgs/validation_images_1280x720/validation/",
        "smpl_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/ground_truth/",
        "mask_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/masks/validation_masks_1280x720/validation",
        "cam_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/cam/validation_annos",
    }


    if len(sys.argv) > 1:
        scene = sys.argv[1]
    else:
        scene = "hdri_50mm"

    if scene == "archviz":
        num_people = "5_10"
    else:
        num_people = "5_15"


    img_config= {
        "split": "validationset",
        "vendor": "renderpeople",
        "dataset": "bfh",
        "scene": scene,
        "num_people": num_people,
        "cam": "cam02",
        "resolution": "1280x720"
    }

    # these are parameters for hdri_50mm, for other scenes see: https://github.com/pixelite1201/agora_evaluation/blob/master/agora_evaluation/projection.py
    if scene == "hdri_50mm":
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 50
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 0
    elif scene == "archviz":
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, 170, 265]
        camYaw = -45
        camPitch = 30

    dslr_sens_width = 36
    dslr_sens_height = 20.25

    #ag_validationset_renderpeople_bfh_hdri_50mm_5_15_00257_1280x720.png
    #ag_validationset_renderpeople_bfh_hdri_50mm_5_15_00148_1280x720.png
    if scene == "hdri_50mm":
        scene_length = 258
    elif scene == "archviz":
        scene_length = 259


    #img_list = [img_filename for img_filename in os.listdir(dataset_config["img_folder_path"]) if img_config["scene"] in img_filename ]
    #scene_length = len(img_list)


    part_segm_filepath = "/home/tuba/Documents/emre/thesis/occlusionIndex/smpl_vert_segmentation.json"

    face_classes, vertex_class_names, faces_per_class = vertex_classes(part_segm_filepath=part_segm_filepath)
    num_joints = len(faces_per_class.keys())

    # flags
    scale = 1

    draw = True

    duration = []
    
    for frame_id in range(196,scene_length):
        #frame_id = 39 # to test only one image
        start_time = time.time()
        img_path, img_filename = construct_img_path(dataset_config, img_config, frame_id=frame_id)

        remaining_secs = np.mean(duration)*(scene_length-frame_id)
        print("%{:.2f} Processing {}... ETA: {:.0f}mins {:.0f}secs".format(frame_id*100/scene_length, img_filename, remaining_secs//60, remaining_secs%60))

        try: # continue if that img does not exists
            img = read_im(img_path, scale, show=False)
            image_height, image_width = img.shape[:2]
        except:
            print("Image does not exists")
            continue
        
        annotation = read_anno(dataset_config=dataset_config, img_filename=img_filename) # remove _1280x720 part
        smpl_paths = annotation["smpl_path"]
        location = annotation["location"]
        cam_extrinsics = annotation["cam_extrinsics"]

        cam_extrinsics = construct_extrinsic_matrix(parameters=cam_extrinsics) 

        num_model = len(smpl_paths)

        vertices_list = []
        faces_list = []


        imgWidth = image_width
        imgHeight = image_height

        image_points_list = []
        faces_list = []

        for model_id, path in enumerate(smpl_paths):

            with open(dataset_config["smpl_folder_path"] + path[:-3] + "pkl", "rb") as pkl_file:
                pkl = pickle.load(pkl_file)

            vertices = pkl["vertices"][0].cpu().detach().numpy()
            faces = pkl["faces"]
            faces_list.append(pkl["faces"])

            vertices3d = pkl["vertices"].cpu().detach().numpy()[0] # pkl["joints"]
            trans3d = location[model_id][:-1]

            gt2d, gt3d_camCoord = project2d(vertices3d, focalLength=focalLength, scene3d=scene3d,
                        trans3d=trans3d,
                        dslr_sens_width=dslr_sens_width,
                        dslr_sens_height=dslr_sens_height,
                        camPosWorld=camPosWorld,
                        cy=imgHeight / 2,
                        cx=imgWidth / 2,
                        imgPath=img_path,
                        yawSMPL=location[model_id][-1],
                        ground_plane=ground_plane,
                        debug_path="./agora/debug_agora_projection",
                        debug=False,
                        ind=-1,
                        pNum=-1,
                        meanPose=False, camPitch=camPitch, camYaw=camYaw)
            
            image_points_list.append(gt2d)
            vertices_list.append(gt3d_camCoord)


        ##############################################################################################################################################
        # initialize canvas and z-buffer
        scene_silhouette_canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        model_silhouette_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)
        
        # body segments
        model_full_segment_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)
        model_occluded_segment_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)
        model_visible_segment_canvas = np.zeros((num_model, image_height, image_width, 3), dtype=np.uint8)

        # per segment 
        per_segment_visible_canvas = np.zeros((num_model, num_joints, image_height, image_width, 3 ), dtype=np.uint8)

        
        # z-buffer for scene silhouettes
        z_buffer = np.ones((image_height, image_width), dtype=np.float64) * np.inf

        # detect truncations
        truncations = np.zeros((num_model, num_joints)) 

        ##############################################################################################################################################
        if verbose:
            silhouette_render_start = time.time()

        # render meshes with z buffer
        for model_id in range(num_model):
            #print("Rendering Model {}".format(model_id))

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

            model_visible_segment_canvas[model_id] = model_full_segment_canvas[model_id].copy()
            model_visible_segment_canvas[model_id][~mask] = 0

        
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

                # there is someone at the scene but non-truncated part is completely occluded.  
                if unique_color not in scene_silhouette_pixel_counter.keys(): # completely occluded
                    model_regional_occlusion = 100
                else:
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

            # the following line assumes all segments have equal sizes, but they dont!
            model_occlusion_statistics["truncation"]["overall"] = np.mean(truncations[model_id])

            ############################################################################################
            if verbose:
                body_segment_calculation_start = time.time()
                print("Silhouette calculations took {} secs".format(body_segment_calculation_start - silhouette_calculation_start))

            # calculate occlusion per body segment
            model_full_segment_counter = count_pixels(mask=model_full_segment_canvas[model_id])
            model_occluded_segment_counter = count_pixels(mask=model_occluded_segment_canvas[model_id])

            for body_segment_color in list(model_occluded_segment_counter.keys()):
                if model_occlusion_statistics["occlusion"]["overall"] == 100:
                    model_occlusion_statistics["occlusion"][joint_name] = 100
                    continue

                total_pixels = model_full_segment_counter[body_segment_color]

                occluded_pixels = model_occluded_segment_counter[body_segment_color]

                body_segment_occlusion_rate = (occluded_pixels / total_pixels) * 100

                joint_name = vertex_class_names[distinct_colors.index(body_segment_color)]
                
                model_occlusion_statistics["occlusion"][joint_name] = body_segment_occlusion_rate

            ############################################################################################
            if verbose:
                self_occlusion_calculation_start = time.time()
                print("Body segment calculations took {} secs".format(self_occlusion_calculation_start - body_segment_calculation_start))
        ############################################################################################

            frame_occlusion_statistics[model_id] = model_occlusion_statistics


        ##############################################################################################################################################
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

                model_full_segment = add_padding(image=model_full_segment, left_padding=5, right_padding=5)
                model_occluded_segment = add_padding(image=model_occluded_segment, left_padding=5, right_padding=5)

                segment_canvas_list.append(model_full_segment)
                segment_canvas_list.append(model_occluded_segment)

            segment_concat_img = np.concatenate(tuple(segment_canvas_list), axis=1)
            
            if visualize:
                cv2.imshow('segment concat image', segment_concat_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        ##############################################################################################################################################
        # save
        save_path = "./agora/regional_occlusion/{}".format(scene)
        os.makedirs(save_path, exist_ok=True)
        image_name = img_filename[:-4]

        if save_vis:
            # save generated mask

            cv2.imwrite("{}/{}.jpg".format(save_path, image_name), scene_silhouette_canvas)
            cv2.imwrite("{}/{}_silhouettes.jpg".format(save_path, image_name), silhouette_concat_img)
            cv2.imwrite("{}/{}_segments.jpg".format(save_path, image_name), segment_concat_img)
            
        with open("{}/{}.json".format(save_path, image_name), "w+") as occlusion_filepointer:
            json.dump(frame_occlusion_statistics, occlusion_filepointer)

        end_time = time.time()
        duration.append(end_time-start_time)
