import os, sys, pickle, time, math
from visualize import *
from utils import *
import trimesh
from occlusion_tools import occlusion_index_new, crowd_metric 
from agora_projection import project2d


save_vis = True
visualize = False
self_occlusion = False
verbose = False

color_list = generate_distinct_colors(15)


if __name__ == '__main__':

    # modify this paths accordingly
    dataset_config = {
        "img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/imgs/validation_images_1280x720/validation/",
        "smpl_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/ground_truth/",
        "mask_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/masks/validation_masks_1280x720/validation",
        "cam_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/data/cam/validation_annos",
    }

    joints = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', "left_hand", "right_hand"
    ]
    num_joints = 24


    if len(sys.argv) > 1:
        scene = sys.argv[1]
    else:
        scene = "hdri_50mm"

    if scene == "archviz":
        num_people = "5_10"
    else:
        num_people = "5_15"

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

    img_config= {
        "split": "validationset",
        "vendor": "renderpeople",
        "dataset": "bfh",
        "scene": scene,
        "num_people": num_people,
        "cam": "cam02",
        "resolution": "1280x720"
    }

    if scene == "hdri_50mm":
        scene_length = 258
    elif scene == "archviz":
        scene_length = 259

    #img_list = [img_filename for img_filename in os.listdir(dataset_config["img_folder_path"]) if img_config["scene"] in img_filename ]
    #scene_length = len(img_list)

    # flags
    scale = 1

    draw = False

    duration = []
    
    for frame_id in range(196, scene_length):
        #frame_id = 39 # to test only one image
        start_time = time.time()
        img_path, img_filename = construct_img_path(dataset_config, img_config, frame_id=frame_id)
        print()

        remaining_secs = np.mean(duration)*(scene_length-frame_id)
        print("%{:.2f} Processing {}... ETA: {:.0f}mins {:.0f}secs".format(frame_id*100/scene_length, img_filename, remaining_secs//60, remaining_secs%60))

        try: # continue if that img does not exists
            img = read_im(img_path, scale, show=False)
            image_height, image_width = img.shape[:2]
        except:
            print("Image does not exists")
            continue

        #################################################################################################################################33
        # calculate 2d locations
        
        annotation = read_anno(dataset_config=dataset_config, img_filename=img_filename) # remove _1280x720 part
        keypoints2d = np.array(annotation["keypoints2d"])[:, :24] * (720/2160) 
        num_models = len(keypoints2d)

        
        #################################################################################################################################
        # calculate occlusion index
        # get masks
        mask_files = construct_mask_path(dataset_config, img_config, frame_id=frame_id)

        all_mask = read_im(mask_files[0], scale, show=False)

        assert img.shape == all_mask.shape, "mask & img shape does not match"     

        # check consistency
        mask_num_models = len(mask_files) - 1

        assert mask_num_models == num_models,  "mask and keypoint model number does not match"

        occlusion_status, truncation_status = occlusion_index_new(masks=mask_files, keypoints=keypoints2d)
        
        #################################################################################################################################
        # calculate crowd index
        # get bboxes

        crowd_index = {
            "overall": 0.0
        }
        crowd_indices = []

        for model_id in range(num_models):
            top_left, bottom_right = approximate_bb(keypoints=keypoints2d[model_id])
            bbox = [top_left, bottom_right]
            current_crowd_index = crowd_metric(bbox=bbox, keypoints=keypoints2d, current_model_id=model_id)
            crowd_index[model_id] = current_crowd_index
            crowd_indices.append(current_crowd_index)

        crowd_index["overall"] = np.mean(crowd_indices)
        
        #################################################################################################################################
        # collect statistics
        frame_occlusion_index = {}
        
        frame_overall_occlusion = np.mean(occlusion_status.astype(int))
        
        frame_occlusion_index["overall"] = frame_overall_occlusion

        per_model_occlusion = []

        for model_id in range(num_models):

            model_occlusion_index = {
                "occlusion": {
                    "overall": np.mean(occlusion_status[model_id].astype(int))
                },
                "truncation": {
                    "overall": np.mean(truncation_status[model_id].astype(int))
                }
            }

            per_model_occlusion.append(model_occlusion_index["occlusion"]["overall"])

            for keypoint_id in range(num_joints):
                model_occlusion_index["occlusion"][joints[keypoint_id]] = bool(occlusion_status[model_id][keypoint_id])
                model_occlusion_index["truncation"][joints[keypoint_id]] = bool(truncation_status[model_id][keypoint_id])

            frame_occlusion_index[model_id] = model_occlusion_index

        #################################################################################################################################
        # save results
        results_dir_oi = "./agora/occlusion_index/{}".format(scene)
        results_dir_ci = "./agora/crowd_index/{}".format(scene)

        os.makedirs(results_dir_oi, exist_ok=True)
        os.makedirs(results_dir_ci, exist_ok=True)

        with open("{}/{}.json".format(results_dir_oi, img_filename[:-4]), "w+") as occlusion_filepointer:
            json.dump(frame_occlusion_index, occlusion_filepointer)

        with open("{}/{}.json".format(results_dir_ci, img_filename[:-4]), "w+") as crowd_filepointer:
            json.dump(crowd_index, crowd_filepointer)

        #################################################################################################################################
        # drawing

        if draw:
            
            # Draw elements
            colors = np.random.randint(255, size=(num_models, 3)).tolist()  # each person has its own color

            bboxes = []
            for model_id in range(num_models):
                bboxes.append(approximate_bb(keypoints=keypoints2d[model_id]))

            # Draw occlusion image
            occlusion_img = draw_masks(img.copy(), mask=all_mask)
            occlusion_img = draw_keypoints_new(occlusion_img, keypoints=keypoints2d, occlusion_index=frame_occlusion_index, kp_names=joints, only_occluded=False, kp_color=(0,0,255))
            occlusion_img = draw_bboxes(occlusion_img, bboxes=bboxes, colors=color_list, index_values=per_model_occlusion)
            

            crowd_img = draw_keypoints_new(img.copy(), keypoints=keypoints2d, occlusion_index=frame_occlusion_index, kp_names=joints, only_occluded=False, kp_color=(0,0,255))
            crowd_img = draw_bboxes(crowd_img, bboxes=bboxes, colors=color_list, index_values=crowd_indices)

            #show_im(occlusion_img)
            #show_im(crowd_img)

            cv2.imwrite("{}/{}".format(results_dir_oi, img_filename), occlusion_img)
            cv2.imwrite("{}/{}".format(results_dir_ci, img_filename), crowd_img)

        end_time = time.time()

        duration.append(end_time-start_time)
        