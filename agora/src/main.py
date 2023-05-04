import os, sys, pickle, time
import numpy as np
import cv2
import pandas as pd
import re
from visualize import *
from utils import *

if __name__ == '__main__':
    validation  = False

    if validation:

        dataset_config = {
            #"img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/image/train_images_1280x720_0/train_0",
            "img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/image/validation_images_1280x720/validation",
            "smpl_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/smpl/",
            #"mask_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/mask/train_masks_1280x720/train",
            "mask_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/mask/validation_masks_1280x720/validation",
            "cam_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/cam"
        }
    else:
        dataset_config = {
            "img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/image/train_images_1280x720_0/train_0",
            # "img_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/image/validation_images_1280x720/validation",
            "smpl_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/smpl/",
            "mask_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/mask/train_masks_1280x720/train",
            # "mask_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/mask/validation_masks_1280x720/validation",
            "cam_folder_path": "/home/tuba/Documents/emre/thesis/dataset/agora/cam"
        }

    """
        split: trainset/validationset, 
        vendor: 3dpeople/axyz/humanalloy/renderpeople, 
        dataset: bfh/body, 
        scene: 
            archviz, #259
            brushifyforest, #258 
            brushifygrasslands, #259
            construction, # 256
            flowers, # 259
            hdri_50mm  # 258
        number of person: 5_10, 5_15, 
        cam: cam00, cam01, cam02, cam03 (only for archviz scene)
    """

    if len(sys.argv) > 1:
        scene = sys.argv[1]
    else:
        scene = "hdri_50mm"

    if scene == "archviz":
        num_people = "5_10"
    else:
        num_people = "5_15"

    if validation:
        img_config= {
            "split": "validationset",
            "vendor": "renderpeople",
            "dataset": "bfh",
            "scene": scene,
            "num_people": num_people,
            "cam": "cam02",
            "resolution": "1280x720"
        }
    else:
        img_config = {
            "split": "trainset",
            "vendor": "renderpeople",
            "dataset": "bfh",
            "scene": scene,
            "num_people": num_people,
            "cam": "cam02",
            "resolution": "1280x720"
        }

    crowd_index = []
    modified_crowd_index = []
    scale = 1
    vis_scale = 0.5
    debug = True

    if validation:
        if img_config["scene"] == "archviz":
            scene_length = 259
        elif img_config["scene"] == "brushifyforest":
            scene_length = 258
        elif img_config["scene"] == "brushifygrasslands":
            scene_length = 259
        elif img_config["scene"] == "construction":
            scene_length = 256
        elif img_config["scene"] == "flowers":
            scene_length = 259
        elif img_config["scene"] == "hdri_50mm":
            scene_length = 258
        else:
            sys.exit()
    else:
        scene_length = 1250

    with open("./{}_result.txt".format(img_config["scene"]), "w+") as result_file:
        pass # clear result file


    dilation_kernel = np.ones((2,2), np.uint8)

    for frame_id in range(362, 363):
        # get img
        img_path, img_filename = construct_img_path(dataset_config, img_config, frame_id=frame_id)

        try: # drop if that img does not exists
            img = show_im(img_path, scale, show=False)
        except:
            continue

        # get masks
        mask_files = construct_mask_path(dataset_config, img_config, frame_id=frame_id)

        all_mask = show_im(mask_files[0], scale, show=False)

        assert img.shape == all_mask.shape, "mask & img shape does not match"

        # get params
        #cam = get_cam(dataset_config["cam_folder_path"])

        # get smpls
        #smpls = list(cam[cam["imgPath"] == img_filename]["gt_path_smpl"])[0]

        #betas, joints = get_smpl(dataset_config["smpl_folder_path"], smpls)

        try:
            # get joints
            if validation:
                joints2d, joints3d = get_joints(img_filename, scale, validation = True)
            else:
                joints2d, joints3d = get_joints(img_filename, scale, validation=False)
        except:
            continue

        coco_joints = convert_joints2coco(joints2d)

        joints2d = coco_joints

        #keypoints_converted = keypoint_type_conversion(joints2d, scale)

        mask_num_models = len(mask_files) - 1

        num_models = joints2d.shape[0]

        assert mask_num_models == num_models, "mask and keypoint model number does not match"

        bboxes = []

        for model_id in range(num_models):
            bbox = approximate_bb(joints2d[model_id], scale)
            bboxes.append(bbox)

        #draw_keypoints_show(img, joints2d, bboxes, scale)
        #draw_keypoints_mask(all_mask, joints2d, scale=1)

        frame_intersection_keypoints = []
        mask_frame_intersection_keypoints = []

        frame_crowd_index = []
        modified_frame_crowd_index = []

        for model_id in range(num_models):

            model_crowd_count = 0

            person_mask = show_im(mask_files[model_id + 1], scale, show=False)  # mask_files[0] shows all the people

            person_unique_color, all_mask_unique_color = get_unique_color(person_mask, all_mask)

            if person_unique_color == None:
                continue

            all_mask[all_mask == (0, 0, 255)] = 0
            dilated_all_mask = cv2.dilate(all_mask, dilation_kernel, iterations=1)

            #mask_count, mask_intersection_keypoints = modified_crowd_metric(all_mask, all_mask_unique_color, joints2d[model_id], drop_object_occlusion=True)
            mask_count, mask_intersection_keypoints = modified_crowd_metric(dilated_all_mask, all_mask_unique_color, joints2d[model_id], drop_object_occlusion=True)

            modified_frame_crowd_index.append(mask_count / joints2d.shape[1])

            if mask_count > 0:
                mask_frame_intersection_keypoints += mask_intersection_keypoints

            for other_model_id in range(num_models):
                if model_id == other_model_id:
                    continue

                count, intersection_keypoints = crowd_metric(bboxes[model_id], joints2d[other_model_id])

                model_crowd_count += count

                if count > 0:
                    frame_intersection_keypoints += intersection_keypoints

            frame_crowd_index.append(model_crowd_count / joints2d.shape[1])  # overlapping / self.keypoints (45)

        #print("Frame points", mask_frame_intersection_keypoints)
        avg_frame_crowd_index = np.mean(frame_crowd_index)  # average over models in the image
        avg_modified_frame_crowd_index = np.mean(modified_frame_crowd_index)

        print("Crowd Index of Frame {} is {}".format(frame_id, avg_frame_crowd_index))
        print("Keypoint occlusion ratio of Frame {} is {}".format(frame_id, avg_modified_frame_crowd_index))

        with open("./{}_result.txt".format(img_config["scene"]), "a+") as result_file:
            result_file.write("{} -> CI: {}, CM:{} \n".format(img_path[41:], avg_frame_crowd_index, avg_modified_frame_crowd_index))


        if debug:
            bbox_keypoints = draw_crowd_keypoints(img, np.array([frame_intersection_keypoints]), bboxes, frame_crowd_index, scale, show=False)
            #mask_keypoints = draw_mask_crowd_keypoints(all_mask, np.array([mask_frame_intersection_keypoints]), bboxes, modified_frame_crowd_index, scale, show=False)
            mask_keypoints = draw_mask_crowd_keypoints(dilated_all_mask, np.array([mask_frame_intersection_keypoints]), bboxes, modified_frame_crowd_index, scale, show=False)

            draw_sidebyside(bbox_keypoints, mask_keypoints, vis_scale)

        crowd_index.append(avg_frame_crowd_index)
        modified_crowd_index.append(avg_modified_frame_crowd_index)

    with open("./{}_result.txt".format(img_config["scene"]), "a+") as result_file:
        result_file.write("Avg CI: {} | Avg CM: {}".format(np.mean(crowd_index), np.mean(modified_crowd_index)))

    print(np.mean(crowd_index))
    print(np.mean(modified_crowd_index))







