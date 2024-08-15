#source ../../../../../smpl_venv/bin/activate
import sys

import cv2, json, os
from vis import *
import numpy as np
from utils import *
from operator import itemgetter
from occlusion_tools import occlusion_index
from smpl_tools import * # import by name 
np.random.seed(39)
from smpl_np_romp import SMPLModel


colors = [
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255]
]

draw_and_save = True

def main():
    prefix = "/home/tuba/Documents/emre/thesis/dataset/ochuman/OCHuman-20230218T083152Z-001/OCHuman/"
    images_prefix = prefix + "images/"
    train_anno_path = "/home/tuba/Documents/emre/thesis/dataset/ochuman/OCHuman-20230218T083152Z-001/OCHuman/ochuman.json"
    data = get_annos(train_anno_path)

    if not "gtSMPL" in os.listdir("./ochuman"):
        process_smpl(read_smpl(prefix=prefix, split="train"))
        process_smpl(read_smpl(prefix=prefix, split="test"))

    num_frames = len(data["images"])

    max_ppl = 0
    
    for frame_id in range(num_frames):

        img_path, img_masks, img_keypoints = ochuman_annotation(data, frame_id=frame_id)
        num_smpl, smpls = get_smpl(img_path)

        if img_path == None: # if there is missing mask or pose of a model
            continue

        #if num_smpl == None:
            #print("Frame {} No SMPL data, skipping...".format(frame_id))
        #    continue

        #if len(img_keypoints) != num_smpl:
            #print("Frame {} SMPL does not match keypoints... Skipping...".format(frame_id))
        #    continue

        
        if not "000414" in img_path:
           continue


        
        num_models = len(img_keypoints)
        color_list = generate_distinct_colors(num_models)

        if num_models > max_ppl:
            max_ppl = num_models

        if num_models > 2:
            print(img_path)


        #################################################################################################################################
        # calculate crowd index
        # get bboxes

        crowd_index_stat = {
            "overall": 0.0
        }
        crowd_indices = []

        for model_id in range(num_models):
            top_left, bottom_right = estimate_bbox(keypoints=img_keypoints[model_id])
            bbox = [top_left, bottom_right]
            current_crowd_index = crowd_metric(bbox=bbox, keypoints=img_keypoints, current_model_id=model_id)
            crowd_index_stat[model_id] = current_crowd_index
            crowd_indices.append(current_crowd_index)

        crowd_index_stat["overall"] = np.mean(crowd_indices)
        #################################################################################################################################

        # Calculate occlusion index
        occlusion_status, missing_status = occlusion_index(masks=img_masks, keypoints=img_keypoints)
        frame_occlusion_index = np.mean(occlusion_status.astype(int))

        # collect statistics
        frame_occlusion_index = {}
        
        frame_overall_occlusion = np.mean(occlusion_status.astype(int))
        
        frame_occlusion_index["overall"] = frame_overall_occlusion

        per_model_occlusion = []

        for model_id in range(num_models):

            model_occlusion_index = {
                "occlusion": {
                    "overall": np.mean(occlusion_status[model_id].astype(int))
                }
            }

            per_model_occlusion.append(model_occlusion_index["occlusion"]["overall"])

            frame_occlusion_index[model_id] = model_occlusion_index

        #################################################################################################################################
        # save results
        results_dir_oi = "./ochuman/occlusion_index"
        results_dir_ci = "./ochuman/crowd_index"

        os.makedirs(results_dir_oi, exist_ok=True)
        os.makedirs(results_dir_ci, exist_ok=True)

        with open("{}/{}.json".format(results_dir_oi, img_path[:-4]), "w+") as occlusion_filepointer:
            json.dump(frame_occlusion_index, occlusion_filepointer)

        with open("{}/{}.json".format(results_dir_ci, img_path[:-4]), "w+") as crowd_filepointer:
            json.dump(crowd_index_stat, crowd_filepointer)

        #################################################################################################################################

        if draw_and_save:
            
            num_models = len(img_keypoints)

            # Draw elements
            #colors = np.random.randint(255, size=(num_models, 3)).tolist()  # each person has its own color

            bboxes = []
            for model_id in range(num_models):
                bboxes.append(estimate_bbox(keypoints=img_keypoints[model_id]))


            img = read_img(images_prefix+img_path, show=False)

            # Draw occlusion image
            occlusion_img = draw_keypoints(img, keypoints=img_keypoints, occlusion_status=occlusion_status,only_occluded=False)
            occlusion_img = draw_masks(occlusion_img, masks=img_masks, colors=colors)
            occlusion_img = draw_bboxes(occlusion_img, bboxes=bboxes, colors=colors, index_values=per_model_occlusion) # TODO

            # Draw crowd image
            crowd_img = draw_keypoints(img, keypoints=img_keypoints, occlusion_status=occlusion_status,only_occluded=False)
            crowd_img = draw_bboxes(crowd_img, bboxes=bboxes, colors=colors, index_values=crowd_indices) # TODO

            # Concat both
            concat_img = draw_sidebyside(img1=crowd_img, img2=occlusion_img, vis_scale=1, show=False)

            #cv2.imwrite("ochuman/processed_imgs_new/{}_crowd.jpg".format(3274), crowd_img)
            save_processed_img(concat_img, img_path)

        

if __name__ == '__main__':
    main()
    #            if cv2.pointPolygonTest(np_obj, (x, y), False) == 1:




